# decon_wrapper.py
# Dask-orchestrated GPU deconvolution with blind PSF estimation.
#
# PSF resolution:
#   Generate a theoretical Gibson-Lanni PSF from the optical parameters, use it
#   only as the starting guess for chunked MATLAB deconvblind, then merge the
#   recovered per-chunk blind PSFs with SNR weighting and save estimated_psf.tif.
#
# Deconvolution:
#   pycudadecon (TemporaryOTF + RLContext) processes each TIFF as full-Z
#   XY chunks using map_overlap.  The requested chunk_xy is treated as the
#   core tile size; <=0 auto-sizes from available VRAM.

import argparse
import re
import tempfile
import time
from glob import glob
from pathlib import Path

import dask.array as da
import numpy as np
from pycudadecon import TemporaryOTF, RLContext, rl_decon
from tifffile import imwrite

from psf_estimation import (
    DEFAULT_BLIND_CHUNK_XY,
    DEFAULT_BLIND_Z_SLICES,
    DEFAULT_SNR_WEIGHT_CAP,
    estimate_psf_from_chunks,
    generate_theoretical_psf,
    open_tiff_memmap,
    resolve_dxy,
    resolve_chunk_xy,
)


# ---------------------------------------------------------------------------
# Dask worker
# ---------------------------------------------------------------------------

CHANNEL_TIMEPOINT_RE = re.compile(r"^CH(?P<channel>\d+)_(?P<timepoint>\d+)(?:_registered_consistent)?$")


def _write_tiff_near_input_or_cwd(path: Path, data: np.ndarray) -> Path:
    try:
        imwrite(str(path), data)
        return path
    except OSError as exc:
        fallback = Path.cwd() / path.name
        print(
            f"  WARNING: cannot write {path}: {exc}; writing {fallback} instead",
            flush=True,
        )
        imwrite(str(fallback), data)
        return fallback


def _parse_int_filter(value: str | None) -> set[int] | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    tokens = [token for token in re.split(r"[\s,]+", text) if token]
    return {int(token) for token in tokens}


def _filter_tiffs(
    tiff_list: list[str],
    channels: set[int] | None,
    timepoints: set[int] | None,
) -> list[str]:
    filtered = []
    for tiff_path in tiff_list:
        match = CHANNEL_TIMEPOINT_RE.match(Path(tiff_path).stem)
        if not match:
            continue
        channel = int(match.group("channel"))
        timepoint = int(match.group("timepoint"))
        if channels is not None and channel not in channels:
            continue
        if timepoints is not None and timepoint not in timepoints:
            continue
        filtered.append(tiff_path)
    return filtered

def _format_seconds(seconds: float) -> str:
    return f"{seconds:.2f}s"


def _chunk_progress(block_info: dict | None, total_chunks: int) -> tuple[int, str]:
    if not isinstance(block_info, dict) or None not in block_info:
        return 0, "unknown"

    info = block_info[None]
    location = info.get("chunk-location")
    num_chunks = info.get("num-chunks")
    if not location or not num_chunks:
        return 0, "unknown"

    chunk_index = 1
    stride = 1
    for loc, count in zip(reversed(location), reversed(num_chunks)):
        chunk_index += loc * stride
        stride *= count

    return chunk_index, f"{chunk_index}/{total_chunks}"


def _decon_chunk(
    chunk: np.ndarray,
    otf_path: str,
    dz: float,
    dxy: float,
    n_iters: int,
    total_chunks: int,
    block_info: dict | None = None,
) -> np.ndarray:
    """
    Process one spatial chunk with pycudadecon.
    Each call opens and closes its own RLContext so chunks can be dispatched
    sequentially without GPU context leakage.
    """
    _, chunk_label = _chunk_progress(block_info, total_chunks)
    if chunk.size == 0:
        return chunk

    print(
        f"  Chunk {chunk_label} started: shape={chunk.shape}, iterations={n_iters}",
        flush=True,
    )

    start = time.perf_counter()
    with RLContext(
        chunk.shape,
        otf_path,
        dzdata=dz,
        dxdata=dxy,
        dzpsf=dz,
        dxpsf=dxy,
    ) as ctx:
        result = rl_decon(chunk, output_shape=ctx.out_shape, n_iters=n_iters)
    elapsed = time.perf_counter() - start
    avg_iter = elapsed / n_iters if n_iters > 0 else elapsed

    print(
        f"  Iteration {n_iters}/{n_iters} of chunk {chunk_label} completed: "
        f"chunk_time={_format_seconds(elapsed)}, "
        f"avg_iteration_time={_format_seconds(avg_iter)}",
        flush=True,
    )
    return np.clip(result, 0, 65535).astype(np.uint16)


def _match_input_intensity_range(output: np.ndarray, input_volume: np.ndarray) -> np.ndarray:
    """Map deconvolved output to the original TIFF intensity range."""
    in_min = float(np.min(input_volume))
    in_max = float(np.max(input_volume))
    out_min = float(np.min(output))
    out_max = float(np.max(output))
    dtype_max = float(np.iinfo(np.uint16).max)

    if not np.isfinite([in_min, in_max, out_min, out_max]).all():
        raise ValueError("Cannot rescale deconvolution output with non-finite intensity bounds")

    if out_max > out_min and in_max > in_min:
        scaled = output.astype(np.float32, copy=False)
        scaled = (scaled - out_min) / (out_max - out_min)
        scaled = scaled * (in_max - in_min) + in_min
    else:
        scaled = output

    return np.clip(np.rint(scaled), 0, dtype_max).astype(np.uint16)


# ---------------------------------------------------------------------------
# Per-TIFF deconvolution
# ---------------------------------------------------------------------------

def _psf_overlap_xy(psf: np.ndarray) -> int:
    """Use a moderate PSF-support halo at chunk boundaries."""
    psf_xy = max(psf.shape[-2:])
    return min(48, max(16, int(np.ceil(psf_xy / 4))))


def deconvolve_tiff(
    image_path: Path,
    psf: np.ndarray,
    n_iters: int,
    dz: float,
    dxy: float,
    wavelength: float,
    na: float,
    ni: float,
    chunk_xy: int = 0,
    vram_gb: float | None = None,
    decon_workers: int = 1,
    overlap_xy: int = 0,
) -> np.ndarray:
    """
    Deconvolve a single TIFF using the supplied PSF.

    Chunks are full-Z XY tiles with PSF-dependent XY overlap so tile boundaries
    are invisible in the merged output.  Z is never split.  `chunk_xy` is the
    core tile size; <=0 chooses a VRAM-aware size.
    """
    volume = open_tiff_memmap(image_path)
    if volume.ndim != 3:
        raise ValueError(f"Expected 3-D volume, got shape {volume.shape}")

    original_shape = volume.shape
    overlap_xy = overlap_xy if overlap_xy > 0 else _psf_overlap_xy(psf)
    overlap_xy = min(overlap_xy, max(1, (min(volume.shape[1:]) - 1) // 2))
    decon_workers = max(1, decon_workers)
    core_chunk_xy = resolve_chunk_xy(
        chunk_xy,
        volume.shape,
        volume.dtype,
        overlap_xy=overlap_xy,
        vram_gb=vram_gb,
        workers=decon_workers,
        min_xy=max(128, overlap_xy * 2),
    )
    if core_chunk_xy <= 0:
        raise ValueError(f"Resolved decon chunk size must be positive, got {core_chunk_xy}")

    nz, ny, nx = volume.shape
    lazy = da.from_array(
        volume,
        chunks=(nz, core_chunk_xy, core_chunk_xy),
        asarray=False,
        lock=False,
    )
    total_chunks = int(np.prod(lazy.numblocks))

    print(f"  Deconvolving {image_path.name}  shape={original_shape}", flush=True)
    print(
        f"  Deconvolution chunks: total={total_chunks}, "
        f"core_chunk_shape=(z={nz}, y={core_chunk_xy}, x={core_chunk_xy}), "
        f"psf_overlap_xy={overlap_xy}, image_xy=({ny}, {nx}), "
        f"iterations_per_chunk={n_iters}, workers={decon_workers}",
        flush=True,
    )

    temp_psf = tempfile.NamedTemporaryFile(suffix=".tif", delete=False)
    psf_path = Path(temp_psf.name)
    temp_psf.close()
    try:
        imwrite(str(psf_path), psf.astype(np.float32, copy=False))
        with TemporaryOTF(
            str(psf_path),
            dzpsf=dz,
            dxpsf=dxy,
            wavelength=int(round(wavelength * 1000)),
            na=na,
            nimm=ni,
        ) as otf:
            processed = lazy.map_overlap(
                _decon_chunk,
                depth={0: 0, 1: overlap_xy, 2: overlap_xy},
                boundary="reflect",
                dtype=np.uint16,
                otf_path=otf.path,
                dz=dz,
                dxy=dxy,
                n_iters=n_iters,
                total_chunks=total_chunks,
            )
            scheduler = "threads" if decon_workers > 1 else "single-threaded"
            output = processed.compute(scheduler=scheduler, num_workers=decon_workers)
    finally:
        psf_path.unlink(missing_ok=True)

    output = _match_input_intensity_range(output, volume)
    print(
        f"  Matched deconvolution intensity range to input: "
        f"min={int(output.min())}, max={int(output.max())}",
        flush=True,
    )

    return output


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Dask-orchestrated GPU deconvolution with blind PSF estimation."
    )

    # Required options
    parser.add_argument("--image_path", required=True,
                        help="Directory containing deskewed CH*.tif files.")
    parser.add_argument("--channels", default="",
                        help="Optional channel filter, e.g. '0', '0 1 2', or '0,1,2'.")
    parser.add_argument("--timepoints", default="",
                        help="Optional timepoint filter, e.g. '0', '0 1', or '0,1'.")

    # Blind estimation options
    parser.add_argument("--blind_iters", type=int, default=10,
                        help="deconvblind iterations per chunk during PSF estimation.")
    parser.add_argument("--chunk_xy",    type=int, default=DEFAULT_BLIND_CHUNK_XY,
                        help="XY tile size for blind PSF estimation. <=0 auto-sizes from VRAM.")
    parser.add_argument("--decon_chunk_xy", type=int, default=0,
                        help="Core XY tile size for CUDA deconvolution. <=0 auto-sizes from VRAM.")
    parser.add_argument("--pad_xy",      type=int, default=32,
                        help="XY halo per edge added to each blind PSF chunk (pixels).")
    parser.add_argument("--pad_z",       type=int, default=20,
                        help="Z halo per edge added before MATLAB deconvblind (pixels).")
    parser.add_argument("--blind_workers", type=int, default=1,
                        help="Concurrent MATLAB deconvblind chunks. <=0 uses CPU affinity, falling back to 32.")
    parser.add_argument("--matlab_threads", type=int, default=1,
                        help="Threads per MATLAB deconvblind process; clamped to 1 or 2.")
    parser.add_argument("--matlab_workers", type=int, default=1,
                        help="Concurrent MATLAB deconvblind processes; default 1 avoids MATLAB orchestration hangs.")
    parser.add_argument("--matlab_timeout", type=int, default=1800,
                        help="Seconds before killing one MATLAB deconvblind chunk. <=0 disables.")
    parser.add_argument("--blind_z_slices", type=int, default=DEFAULT_BLIND_Z_SLICES,
                        help="Z planes used per blind PSF tile. <=0 uses full Z.")
    parser.add_argument("--snr_weight_cap", type=float, default=DEFAULT_SNR_WEIGHT_CAP,
                        help="Maximum per-chunk SNR weight before weighted PSF merge; <=0 disables cap.")
    parser.add_argument("--prefetch_chunks", type=int, default=0,
                        help="Number of PSF tiles to keep submitted/read ahead. <=0 uses one worker batch.")
    parser.add_argument("--decon_workers", type=int, default=1,
                        help="Dask workers for CUDA deconvolution chunks.")
    parser.add_argument("--overlap_xy", type=int, default=0,
                        help="Override CUDA decon XY overlap. <=0 uses a capped PSF/4 estimate.")
    parser.add_argument("--vram_gb", type=float, default=None,
                        help="Override detected free VRAM in GiB for auto chunk sizing.")
    parser.add_argument("--cache_dir", default=None,
                        help="Directory for cached blind PSF estimates.")
    parser.add_argument("--no_psf_cache", action="store_true",
                        help="Disable reuse of cached blind PSF estimates.")

    # Deconvolution options
    parser.add_argument("--iter",       type=int,   default=10,
                        help="RL deconvolution iterations.")
    parser.add_argument("--background", type=float, default=0.0,
                        help="Background value to subtract before decon.")

    # Optional optical parameters used to generate the blind-estimation PSF seed.
    parser.add_argument("--na",          type=float, default=1.0,
                        help="Backward-compatible detection numerical aperture.")
    parser.add_argument("--detection_na", type=float, default=None,
                        help="Detection objective numerical aperture. Overrides --na when provided.")
    parser.add_argument("--illumination_na", type=float, default=None,
                        help="Illumination numerical aperture metadata; not used by psfmodels.")
    parser.add_argument("--wavelength",  type=float, default=0.525,
                        help="Emission wavelength in µm.")
    parser.add_argument("--ni",          type=float, default=1.33,
                        help="Refractive index of immersion medium.")
    parser.add_argument("--ns",          type=float, default=None,
                        help="Sample refractive index.")
    parser.add_argument("--ni0",         type=float, default=None,
                        help="Design immersion refractive index.")
    parser.add_argument("--tg",          type=float, default=None,
                        help="Experimental coverslip thickness in µm.")
    parser.add_argument("--tg0",         type=float, default=None,
                        help="Design coverslip thickness in µm.")
    parser.add_argument("--ng",          type=float, default=None,
                        help="Experimental coverslip refractive index.")
    parser.add_argument("--ng0",         type=float, default=None,
                        help="Design coverslip refractive index.")
    parser.add_argument("--ti0",         type=float, default=None,
                        help="Objective working distance in µm.")
    parser.add_argument("--oversample_factor", type=int, default=3,
                        help="PSF model oversampling factor.")
    parser.add_argument("--psf_model", choices=("vectorial", "scalar", "gaussian"), default="vectorial",
                        help="psfmodels PSF model.")
    parser.add_argument("--camera_pixel_size", type=float, default=None,
                        help="Camera pixel size in µm; used to derive dxy when --dxy <= 0.")
    parser.add_argument("--magnification", type=float, default=None,
                        help="Total magnification; used to derive dxy when --dxy <= 0.")
    parser.add_argument("--dxy",         type=float, default=0.1,
                        help="Lateral pixel size in µm.")
    parser.add_argument("--dz",          type=float, default=0.3,
                        help="Axial step size in µm.")
    parser.add_argument("--psf_size_z",  type=int,   default=61,
                        help="Z size of PSF volume.")
    parser.add_argument("--psf_size_xy", type=int,   default=129,
                        help="XY size of PSF volume.")

    # Misc, usually unneeded
    parser.add_argument("--script_dir",  default=str(Path(__file__).parent),
                        help="Directory containing readtiffstack.m / writetiffstack.m.")

    args = parser.parse_args()

    image_dir = Path(args.image_path)

    # Collect all deskewed TIFFs, sorted so index 0 is deterministic
    tiff_list = sorted(
        glob(str(image_dir / "CH*.tif")) +
        glob(str(image_dir / "CH*.tiff"))
    )
    channels = _parse_int_filter(args.channels)
    timepoints = _parse_int_filter(args.timepoints)
    tiff_list = _filter_tiffs(tiff_list, channels, timepoints)
    if not tiff_list:
        print(
            f"Error: no CH*.tif files found in {image_dir} "
            f"matching channels={args.channels!r}, timepoints={args.timepoints!r}"
        )
        raise SystemExit(1)

    print(
        f"Found {len(tiff_list)} TIFF(s) to process "
        f"for channels={args.channels or 'all'}, timepoints={args.timepoints or 'all'}.",
        flush=True,
    )

    # ------------------------------------------------------------------
    # PSF resolution
    # ------------------------------------------------------------------

    dxy = resolve_dxy(args.dxy, args.camera_pixel_size, args.magnification)
    detection_na = args.detection_na if args.detection_na is not None else args.na

    # Build the optical-model PSF seed.  This is intentionally not accepted as
    # the final deconvolution PSF because the measured blind estimates are much
    # closer to the observed data.
    psf_seed = generate_theoretical_psf(
        na=args.na,
        detection_na=args.detection_na,
        illumination_na=args.illumination_na,
        wavelength=args.wavelength,
        ni=args.ni,
        ns=args.ns,
        ni0=args.ni0,
        tg=args.tg,
        tg0=args.tg0,
        ng=args.ng,
        ng0=args.ng0,
        ti0=args.ti0,
        oversample_factor=args.oversample_factor,
        psf_model=args.psf_model,
        dxy=dxy,
        dz=args.dz,
        psf_size_z=args.psf_size_z,
        psf_size_xy=args.psf_size_xy,
        background=args.background,
    )

    print("Running blind PSF estimation on first TIFF...", flush=True)
    psf = estimate_psf_from_chunks(
        image_path=tiff_list[0],
        psf_seed=psf_seed,
        n_iters=args.blind_iters,
        chunk_xy=args.chunk_xy,
        pad_xy=args.pad_xy,
        pad_z=args.pad_z,
        script_dir=args.script_dir,
        max_workers=args.blind_workers,
        prefetch_chunks=args.prefetch_chunks,
        vram_gb=args.vram_gb,
        cache_dir=args.cache_dir,
        use_cache=not args.no_psf_cache,
        matlab_threads=args.matlab_threads,
        matlab_workers=args.matlab_workers,
        matlab_timeout=args.matlab_timeout,
        snr_weight_cap=args.snr_weight_cap,
        blind_z_slices=args.blind_z_slices,
    )
    psf_save_path = image_dir / "estimated_psf.tif"
    psf_save_path = _write_tiff_near_input_or_cwd(psf_save_path, psf)
    print(f"Merged PSF saved to {psf_save_path}", flush=True)

    # ------------------------------------------------------------------
    # Deconvolve all TIFFs with the resolved PSF
    # ------------------------------------------------------------------

    for tiff_path in tiff_list:
        tiff_path = Path(tiff_path)
        output = deconvolve_tiff(
            image_path=tiff_path,
            psf=psf,
            n_iters=args.iter,
            dz=args.dz,
            dxy=dxy,
            wavelength=args.wavelength,
            na=detection_na,
            ni=args.ni,
            chunk_xy=args.decon_chunk_xy,
            vram_gb=args.vram_gb,
            decon_workers=args.decon_workers,
            overlap_xy=args.overlap_xy,
        )

        stem = tiff_path.name.replace(".tiff", "").replace(".tif", "")
        out_name = f"DB2_{stem}.tif" if "CH" in stem else "DB2_deconvolved_output.tif"
        imwrite(out_name, output)
        print(f"  Saved {out_name}", flush=True)

    print("All TIFFs deconvolved.", flush=True)


if __name__ == "__main__":
    main()
