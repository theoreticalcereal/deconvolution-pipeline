# psf_estimation.py
# Blind PSF estimation via chunked MATLAB deconvblind + weighted merge.
#
# Workflow:
#   Memory-map the first deskewed TIFF.
#   Split it into full-Z XY tiles sized from available VRAM unless overridden.
#   Tiles are read ahead, written to temp TIFFs, and sent to MATLAB deconvblind.
#   MATLAB writes back an estimated PSF TIFF per tile.
#   Python collects all per-tile PSFs and returns an SNR-weighted PSF merge.
#
# The returned PSF is float32, normalised to sum=1, and saved as estimated_psf.tif
# next to the input image so pycudadecon can pick it up via TemporaryOTF.

import argparse
import concurrent.futures as futures
import hashlib
import inspect
import json
import math
import os
import subprocess
import tempfile
import sys
import threading
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import psfmodels as pm
from tifffile import TiffFile, imread, imwrite, memmap as tiff_memmap

DEFAULT_CPU_THREADS = 32
DEFAULT_SNR_WEIGHT_CAP = 100.0
DEFAULT_BLIND_MEMORY_MULTIPLIER = 28.0
DEFAULT_BLIND_MEMORY_OVERHEAD_GB = 1.0
DEFAULT_BLIND_Z_SLICES = 128
DEFAULT_BLIND_CHUNK_XY = 256


def _ensure_writable_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        dir=path,
        prefix=".write_test_",
        delete=True,
    ):
        pass


def _resolve_psf_cache_root(image_path: Path, cache_dir: str | Path | None) -> Path:
    if cache_dir:
        cache_root = Path(cache_dir)
        _ensure_writable_dir(cache_root)
        return cache_root

    preferred = image_path.parent / ".psf_cache"
    fallback = Path.cwd() / ".psf_cache"
    for cache_root in (preferred, fallback):
        try:
            _ensure_writable_dir(cache_root)
        except OSError as exc:
            if cache_root == preferred:
                print(
                    f"  WARNING: cannot use PSF cache {cache_root}: {exc}; "
                    f"falling back to {fallback}",
                    flush=True,
                )
            continue
        return cache_root

    raise PermissionError(
        f"Unable to create PSF cache in {preferred} or fallback {fallback}"
    )


# ---------------------------------------------------------------------------
# Theoretical PSF seed for blind deconvolution
# ---------------------------------------------------------------------------

def _available_cpu_threads(default: int = DEFAULT_CPU_THREADS) -> int:
    try:
        return max(1, len(os.sched_getaffinity(0)))
    except (AttributeError, OSError):
        pass
    count = os.cpu_count()
    return count if count and count > 0 else default


def resolve_worker_count(requested_workers: int, default: int = DEFAULT_CPU_THREADS) -> int:
    if requested_workers > 0:
        return requested_workers
    return min(_available_cpu_threads(default=default), default)


def _parse_memory_bytes(value: str | None) -> int | None:
    if not value:
        return None
    text = value.strip().upper()
    if not text:
        return None
    multiplier = 1
    if text.endswith(("K", "KB")):
        multiplier = 1024
        text = text.rstrip("B").rstrip("K")
    elif text.endswith(("M", "MB")):
        multiplier = 1024 ** 2
        text = text.rstrip("B").rstrip("M")
    elif text.endswith(("G", "GB")):
        multiplier = 1024 ** 3
        text = text.rstrip("B").rstrip("G")
    elif text.endswith(("T", "TB")):
        multiplier = 1024 ** 4
        text = text.rstrip("B").rstrip("T")
    try:
        number = float(text)
    except ValueError:
        return None
    # Slurm memory variables without suffix are MB.
    if multiplier == 1 and number < 10_000_000:
        multiplier = 1024 ** 2
    return int(number * multiplier)


def _cgroup_memory_limit_bytes() -> int | None:
    candidates = [
        Path("/sys/fs/cgroup/memory.max"),
        Path("/sys/fs/cgroup/memory/memory.limit_in_bytes"),
    ]
    for path in candidates:
        try:
            text = path.read_text(encoding="utf-8").strip()
        except OSError:
            continue
        if not text or text == "max":
            continue
        try:
            value = int(text)
        except ValueError:
            continue
        if 0 < value < 1 << 60:
            return value
    return None


def _allocated_memory_bytes() -> int | None:
    slurm_node = _parse_memory_bytes(os.environ.get("SLURM_MEM_PER_NODE"))
    if slurm_node:
        return slurm_node

    slurm_per_cpu = _parse_memory_bytes(os.environ.get("SLURM_MEM_PER_CPU"))
    if slurm_per_cpu:
        cpus = int(os.environ.get("SLURM_CPUS_PER_TASK") or _available_cpu_threads())
        return slurm_per_cpu * max(1, cpus)

    cgroup_limit = _cgroup_memory_limit_bytes()
    if cgroup_limit:
        return cgroup_limit

    try:
        import psutil

        return int(psutil.virtual_memory().available)
    except Exception:
        return None


def _blind_chunk_input_bytes(
    volume_shape: tuple[int, int, int],
    dtype: np.dtype,
    chunk_xy: int,
    halo_xy: int,
) -> int:
    nz, ny, nx = volume_shape
    tile_y = min(ny, chunk_xy + 2 * halo_xy)
    tile_x = min(nx, chunk_xy + 2 * halo_xy)
    return int(nz * tile_y * tile_x * np.dtype(dtype).itemsize)


def select_blind_z_window(
    volume: np.ndarray,
    max_z_slices: int = DEFAULT_BLIND_Z_SLICES,
    sample_planes: int = 64,
) -> tuple[slice, str]:
    nz = volume.shape[0]
    if max_z_slices <= 0 or nz <= max_z_slices:
        return slice(None), f"full_z=0:{nz}"

    sample_count = max(1, min(sample_planes, nz))
    sample_indices = np.unique(np.linspace(0, nz - 1, sample_count, dtype=int))
    scores = []
    for z in sample_indices:
        plane = np.asarray(volume[z], dtype=np.float32)
        scores.append(float(np.percentile(plane, 99.9)))

    if max(scores) <= min(scores):
        center_z = nz // 2
        score_detail = "flat_sample_scores"
    else:
        center_z = int(sample_indices[int(np.argmax(scores))])
        score_detail = "brightest_sample"
    start = max(0, center_z - (max_z_slices // 2))
    stop = min(nz, start + max_z_slices)
    start = max(0, stop - max_z_slices)
    return (
        slice(start, stop),
        f"bright_z_window={start}:{stop}, center={center_z}, "
        f"sampled_planes={len(sample_indices)}, selector={score_detail}",
    )


def resolve_blind_worker_count(
    requested_workers: int,
    cpu_workers: int,
    volume_shape: tuple[int, int, int],
    dtype: np.dtype,
    chunk_xy: int,
    halo_xy: int,
    memory_multiplier: float = DEFAULT_BLIND_MEMORY_MULTIPLIER,
    overhead_gb: float = DEFAULT_BLIND_MEMORY_OVERHEAD_GB,
) -> tuple[int, str]:
    if requested_workers > 0:
        return requested_workers, "explicit"

    memory_bytes = _allocated_memory_bytes()
    if not memory_bytes:
        return cpu_workers, "cpu"

    chunk_bytes = _blind_chunk_input_bytes(volume_shape, dtype, chunk_xy, halo_xy)
    per_worker = chunk_bytes * memory_multiplier + overhead_gb * (1024 ** 3)
    usable = memory_bytes * 0.70
    memory_workers = max(1, int(usable // max(1, per_worker)))
    resolved = max(1, min(cpu_workers, memory_workers))
    detail = (
        f"cpu={cpu_workers}, memory_cap={memory_workers}, "
        f"allocated={memory_bytes / (1024 ** 3):.1f}GiB, "
        f"estimated_per_worker={per_worker / (1024 ** 3):.1f}GiB"
    )
    return resolved, detail


def generate_theoretical_psf(
    na: float = 1.0,
    detection_na: float | None = None,
    illumination_na: float | None = None,
    wavelength: float = 0.525,      # µm
    ni: float = 1.33,
    ns: float | None = None,
    ni0: float | None = None,
    tg: float | None = None,
    tg0: float | None = None,
    ng: float | None = None,
    ng0: float | None = None,
    ti0: float | None = None,
    oversample_factor: int = 3,
    psf_model: str = "vectorial",
    dxy: float = 0.1,               # µm, lateral pixel size
    dz: float = 0.3,                # µm, axial step
    psf_size_z: int = 61,
    psf_size_xy: int = 128,
    background: float = 0.0,
) -> np.ndarray:
    """
    Generate a 3-D Gibson-Lanni PSF using psfmodels.

    psfmodels generates the detection PSF.  `illumination_na` is accepted for
    pipeline metadata, but this scalar/vectorial PSF model does not use it.

    Returns float32 array of shape (psf_size_z, psf_size_xy, psf_size_xy),
    background-subtracted and normalised to sum = 1.
    """
    detection_na = detection_na if detection_na is not None else na
    requested_kwargs = {
        "z": psf_size_z,
        "nx": psf_size_xy,
        "dz": dz,
        "dxy": dxy,
        "NA": detection_na,
        "wvl": wavelength,
        "ni": ni,
        "oversample_factor": oversample_factor,
        "model": psf_model,
    }
    optional_kwargs = {
        "ns": ns,
        "ni0": ni0,
        "tg": tg,
        "tg0": tg0,
        "ng": ng,
        "ng0": ng0,
        "ti0": ti0,
    }
    requested_kwargs.update({name: value for name, value in optional_kwargs.items() if value is not None})
    signature = inspect.signature(pm.make_psf)
    missing = [name for name in requested_kwargs if name not in signature.parameters]
    if missing:
        raise RuntimeError(
            "psfmodels.make_psf API mismatch; missing expected parameter(s): "
            + ", ".join(missing)
        )
    psf = pm.make_psf(**requested_kwargs).astype(np.float32)

    psf = np.maximum(psf - background, 0)
    total = psf.sum()
    if total > 0:
        psf /= total
    return psf


def resolve_dxy(
    dxy: float,
    camera_pixel_size: float | None = None,
    magnification: float | None = None,
) -> float:
    if dxy > 0:
        return dxy
    if camera_pixel_size and magnification and camera_pixel_size > 0 and magnification > 0:
        return camera_pixel_size / magnification
    raise ValueError("dxy must be > 0, or camera_pixel_size and magnification must be provided")


# ---------------------------------------------------------------------------
# Per-chunk blind estimation via MATLAB deconvblind
# ---------------------------------------------------------------------------

def _normalise_psf(psf: np.ndarray) -> np.ndarray:
    psf = np.nan_to_num(psf.astype(np.float32, copy=False), nan=0.0, posinf=0.0, neginf=0.0)
    psf = np.clip(psf, 0, None)
    total = float(psf.sum())
    if total > 0:
        psf = psf / total
    return psf.astype(np.float32, copy=False)


def open_tiff_memmap(path: str | Path) -> np.ndarray:
    """
    Return a read-only array-like TIFF volume without forcing a full RAM load.

    `tifffile.memmap` maps compatible contiguous TIFF data directly.  Some TIFFs
    cannot be directly memory-mapped; for those, tifffile can materialise a
    temporary memmap via `asarray(out="memmap")`, which still keeps downstream
    chunking bounded instead of holding the whole image as an ndarray.
    """
    path = Path(path)
    try:
        return tiff_memmap(str(path), mode="r")
    except Exception:
        with TiffFile(str(path)) as tif:
            return tif.asarray(out="memmap")


def detect_vram_bytes() -> int | None:
    """Best-effort free VRAM query using nvidia-smi."""
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=memory.free",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            check=False,
        )
    except OSError:
        return None

    if result.returncode != 0:
        return None

    free_mb = []
    for line in result.stdout.splitlines():
        try:
            free_mb.append(int(line.strip().split()[0]))
        except (IndexError, ValueError):
            continue
    if not free_mb:
        return None
    return min(free_mb) * 1024 * 1024


def resolve_chunk_xy(
    requested_xy: int,
    volume_shape: tuple[int, int, int],
    dtype: np.dtype,
    overlap_xy: int = 0,
    vram_gb: float | None = None,
    workers: int = 1,
    safety_fraction: float = 0.55,
    memory_multiplier: float = 18.0,
    min_xy: int = 128,
    max_xy: int | None = None,
) -> int:
    """
    Resolve an XY chunk size.  Positive `requested_xy` is treated as an explicit
    override; zero or negative values trigger a VRAM-aware estimate.
    """
    if requested_xy > 0:
        return requested_xy

    nz, ny, nx = volume_shape
    max_xy = max_xy or min(ny, nx)
    vram_bytes = int(vram_gb * (1024 ** 3)) if vram_gb and vram_gb > 0 else detect_vram_bytes()
    if not vram_bytes:
        return min(512, max_xy)

    workers = max(1, workers)
    bytes_per_voxel = np.dtype(dtype).itemsize
    usable = vram_bytes * safety_fraction / workers
    denom = max(1, nz) * bytes_per_voxel * memory_multiplier
    overlapped_xy = int(math.sqrt(max(1, usable / denom)))
    core_xy = max(min_xy, overlapped_xy - (2 * overlap_xy))
    core_xy = min(core_xy, max_xy)
    aligned = max(min_xy, (core_xy // 32) * 32)
    return max(32, min(aligned, max_xy))


def _psf_cache_key(
    image_path: Path,
    psf_seed: np.ndarray,
    n_iters: int,
    chunk_xy: int,
    pad_xy: int,
    pad_z: int,
    script_dir: Path,
    merge_mode: str,
    snr_weight_cap: float,
    z_window: tuple[int | None, int | None],
) -> str:
    stat = image_path.stat()
    payload = {
        "image": str(image_path.resolve()),
        "size": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
        "seed_shape": psf_seed.shape,
        "seed_sha256": hashlib.sha256(np.ascontiguousarray(psf_seed).view(np.uint8)).hexdigest(),
        "n_iters": n_iters,
        "chunk_xy": chunk_xy,
        "pad_xy": pad_xy,
        "pad_z": pad_z,
        "script_dir": str(script_dir.resolve()),
        "merge_mode": merge_mode,
        "snr_weight_cap": snr_weight_cap,
        "z_window": z_window,
        "version": 3,
    }
    encoded = json.dumps(payload, sort_keys=True).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()[:20]

def _write_matlab_stack(array: np.ndarray, path: Path, scale_float: bool = False) -> None:
    """Write a stack in a TIFF format that MATLAB's Tiff reader handles reliably."""
    array = np.asarray(array)
    if np.issubdtype(array.dtype, np.floating):
        finite = np.nan_to_num(array, nan=0.0, posinf=0.0, neginf=0.0)
        finite = np.clip(finite, 0, None)
        if scale_float:
            max_value = finite.max()
            if max_value > 0:
                finite = finite / max_value * np.iinfo(np.uint16).max
        array = np.clip(np.rint(finite), 0, np.iinfo(np.uint16).max).astype(np.uint16)
    imwrite(str(path), array)


def _write_chunk(chunk: np.ndarray, path: Path) -> None:
    _write_matlab_stack(chunk, path)


def _run_matlab_deconvblind(
    chunk_path: Path,
    psf_seed: np.ndarray,
    psf_seed_path: Path,
    output_psf_path: Path,
    n_iters: int,
    pad_z: int,
    script_dir: Path,
    matlab_threads: int,
    matlab_timeout: int,
) -> None:
    """
    Call MATLAB deconvblind on one chunk.  The script writes the recovered PSF
    to output_psf_path as a float32 TIFF.

    MATLAB is invoked with -batch so it exits cleanly on completion or error.
    """
    _write_matlab_stack(psf_seed, psf_seed_path, scale_float=True)

    matlab_threads = min(2, max(1, matlab_threads))
    pad_z = max(0, pad_z)
    matlab_thread_cmd = f"maxNumCompThreads({matlab_threads}); "
    z_pad_cmd = (
        f"chunk = padarray(chunk, [0 0 {pad_z}], 'symmetric'); "
        if pad_z > 0 else ""
    )
    matlab_cmd = (
        f"addpath('{script_dir}'); "
        f"{matlab_thread_cmd}"
        f"chunk = single(readtiffstack('{chunk_path}')); "
        f"psf_seed = single(readtiffstack('{psf_seed_path}')); "
        f"psf_seed = psf_seed / sum(psf_seed(:)); "
        f"{z_pad_cmd}"
        f"[~, psf_est] = deconvblind(chunk, psf_seed, {n_iters}); "
        f"psf_est = single(psf_est); "
        f"psf_est = psf_est / sum(psf_est(:)); "
        f"writetiffstack(psf_est, '{output_psf_path}');"
    )
    matlab_args = ["matlab"]
    if matlab_threads == 1:
        matlab_args.append("-singleCompThread")
    matlab_args.extend(["-batch", matlab_cmd])
    env = os.environ.copy()
    for name in (
        "OMP_NUM_THREADS",
        "OMP_THREAD_LIMIT",
        "MKL_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
    ):
        env[name] = str(matlab_threads)

    try:
        result = subprocess.run(
            matlab_args,
            capture_output=True,
            text=True,
            env=env,
            timeout=matlab_timeout if matlab_timeout > 0 else None,
        )
    except subprocess.TimeoutExpired as exc:
        raise RuntimeError(
            f"MATLAB deconvblind timed out for chunk {chunk_path.name} "
            f"after {matlab_timeout}s.\n"
            f"STDOUT: {exc.stdout or ''}\nSTDERR: {exc.stderr or ''}"
        ) from exc
    if result.returncode != 0:
        raise RuntimeError(
            f"MATLAB deconvblind failed for chunk {chunk_path.name} "
            f"(returncode={result.returncode}).\n"
            f"STDOUT: {result.stdout}\nSTDERR: {result.stderr}"
        )


# ---------------------------------------------------------------------------
# Main estimation entry point
# ---------------------------------------------------------------------------

def _tile_origins(ny: int, nx: int, chunk_xy: int) -> list[tuple[int, int, int, int]]:
    min_tile = max(1, chunk_xy // 2)
    origins = []
    for y0 in range(0, ny, chunk_xy):
        for x0 in range(0, nx, chunk_xy):
            y1 = min(y0 + chunk_xy, ny)
            x1 = min(x0 + chunk_xy, nx)
            if (y1 - y0) >= min_tile and (x1 - x0) >= min_tile:
                origins.append((y0, x0, y1, x1))
    return origins


def _extract_tile_with_halo(
    volume: np.ndarray,
    y0: int,
    x0: int,
    y1: int,
    x1: int,
    halo_xy: int,
) -> np.ndarray:
    _, ny, nx = volume.shape
    read_y0 = max(0, y0 - halo_xy)
    read_y1 = min(ny, y1 + halo_xy)
    read_x0 = max(0, x0 - halo_xy)
    read_x1 = min(nx, x1 + halo_xy)

    chunk = np.asarray(volume[:, read_y0:read_y1, read_x0:read_x1])

    before_y = read_y0 - (y0 - halo_xy)
    after_y = (y1 + halo_xy) - read_y1
    before_x = read_x0 - (x0 - halo_xy)
    after_x = (x1 + halo_xy) - read_x1
    if before_y or after_y or before_x or after_x:
        chunk = np.pad(
            chunk,
            pad_width=((0, 0), (before_y, after_y), (before_x, after_x)),
            mode="reflect",
        )
    return chunk


def _snr_weight(core: np.ndarray, weight_cap: float = DEFAULT_SNR_WEIGHT_CAP) -> float:
    sample = np.asarray(core, dtype=np.float32)
    if sample.size == 0:
        return 0.0
    p50, p90, p99 = np.percentile(sample, [50, 90, 99])
    noise_region = sample[sample <= p90]
    if noise_region.size == 0:
        noise_region = sample
    mad = np.median(np.abs(noise_region - np.median(noise_region)))
    noise = max(1.4826 * float(mad), float(np.std(noise_region)), 1.0)
    snr = max(0.0, float(p99 - p50)) / noise
    weight = max(1e-3, snr * snr)
    if weight_cap > 0:
        weight = min(weight, weight_cap)
    return weight


def _format_seconds(seconds: float) -> str:
    return f"{seconds:.2f}s"


def _estimate_one_tile(
    idx: int,
    total_tiles: int,
    volume: np.ndarray,
    tile: tuple[int, int, int, int],
    psf_seed: np.ndarray,
    pad_xy: int,
    pad_z: int,
    n_iters: int,
    script_dir: Path,
    tmpdir: Path,
    matlab_lock: threading.Semaphore,
    matlab_threads: int,
    matlab_timeout: int,
    snr_weight_cap: float,
) -> tuple[int, np.ndarray | None, float, str | None]:
    chunk_start = time.perf_counter()
    y0, x0, y1, x1 = tile
    started_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(
        f"  Blind chunk {idx + 1}/{total_tiles} started at {started_at}: "
        f"tile=({y0}:{y1}, {x0}:{x1})",
        flush=True,
    )
    core = np.asarray(volume[:, y0:y1, x0:x1])
    weight = _snr_weight(core, weight_cap=snr_weight_cap)
    chunk = _extract_tile_with_halo(volume, y0, x0, y1, x1, pad_xy)
    chunk_shape = chunk.shape
    read_elapsed = time.perf_counter() - chunk_start

    chunk_path = tmpdir / f"chunk_{idx:04d}.tif"
    seed_path = tmpdir / f"seed_{idx:04d}.tif"
    psf_out_path = tmpdir / f"psf_out_{idx:04d}.tif"
    write_start = time.perf_counter()
    _write_chunk(chunk, chunk_path)
    del chunk
    write_elapsed = time.perf_counter() - write_start

    try:
        matlab_wait_start = time.perf_counter()
        with matlab_lock:
            matlab_wait_elapsed = time.perf_counter() - matlab_wait_start
            matlab_started_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            print(
                f"  Blind chunk {idx + 1}/{total_tiles} MATLAB started at "
                f"{matlab_started_at}: wait={_format_seconds(matlab_wait_elapsed)}, "
                f"input_shape={chunk_shape}",
                flush=True,
            )
            matlab_start = time.perf_counter()
            _run_matlab_deconvblind(
                chunk_path,
                psf_seed,
                seed_path,
                psf_out_path,
                n_iters,
                pad_z,
                script_dir,
                matlab_threads,
                matlab_timeout,
            )
            matlab_elapsed = time.perf_counter() - matlab_start
    except RuntimeError as exc:
        return idx, None, weight, str(exc)

    if not psf_out_path.exists():
        return idx, None, weight, "MATLAB produced no PSF output"

    output_read_start = time.perf_counter()
    psf_chunk = imread(str(psf_out_path)).astype(np.float32)
    output_read_elapsed = time.perf_counter() - output_read_start
    if psf_chunk.shape != psf_seed.shape:
        return (
            idx,
            None,
            weight,
            f"PSF shape {psf_chunk.shape} != seed shape {psf_seed.shape}",
        )

    total_elapsed = time.perf_counter() - chunk_start
    completed_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(
        f"  Blind chunk {idx + 1}/{total_tiles} completed at {completed_at}: "
        f"tile=({y0}:{y1}, {x0}:{x1}), "
        f"snr_weight={weight:.3g}, "
        f"read={_format_seconds(read_elapsed)}, "
        f"write={_format_seconds(write_elapsed)}, "
        f"matlab_wait={_format_seconds(matlab_wait_elapsed)}, "
        f"matlab={_format_seconds(matlab_elapsed)}, "
        f"output_read={_format_seconds(output_read_elapsed)}, "
        f"total={_format_seconds(total_elapsed)}",
        flush=True,
    )
    return idx, _normalise_psf(psf_chunk), weight, None

def estimate_psf_from_chunks(
    image_path: str | Path,
    psf_seed: np.ndarray,
    n_iters: int = 10,
    chunk_xy: int = DEFAULT_BLIND_CHUNK_XY,
    pad_xy: int = 32,
    pad_z: int = 20,
    script_dir: str | Path | None = None,
    max_workers: int = 0,
    prefetch_chunks: int = 0,
    vram_gb: float | None = None,
    cache_dir: str | Path | None = None,
    use_cache: bool = True,
    matlab_threads: int = 1,
    matlab_workers: int = 1,
    matlab_timeout: int = 1800,
    snr_weight_cap: float = DEFAULT_SNR_WEIGHT_CAP,
    blind_z_slices: int = DEFAULT_BLIND_Z_SLICES,
) -> np.ndarray:
    """
    Estimate a PSF by running MATLAB deconvblind on spatial XY chunks of the
    first deskewed TIFF and merging per-chunk estimates by SNR-weighted mean.

    Parameters
    ----------
    image_path  : path to the deskewed input TIFF (full Z stack, 3-D).
    psf_seed    : initial PSF guess, float32 numpy array (nz_psf, ny_psf, nx_psf).
                  Typically the output of generate_theoretical_psf().
    n_iters     : number of deconvblind iterations per chunk.
    chunk_xy    : XY tile size.  <= 0 chooses a VRAM-aware size.
    pad_xy      : XY halo per edge before deconvblind. Interior tiles include
                  real neighboring pixels; only image borders are reflect-padded.
    pad_z       : Z halo per edge before deconvblind, applied symmetrically in MATLAB.
    script_dir  : directory containing readtiffstack.m / writetiffstack.m.
                  Defaults to the directory of this script.

    Returns
    -------
    float32 numpy array of shape matching psf_seed, normalised to sum = 1.
    """
    image_path = Path(image_path)
    script_dir = Path(script_dir) if script_dir else Path(__file__).parent

    print(f"Memory-mapping {image_path} for PSF estimation...", flush=True)
    volume = open_tiff_memmap(image_path)  # (nz, ny, nx)
    if volume.ndim != 3:
        raise ValueError(f"Expected a 3-D volume, got shape {volume.shape}")

    original_shape = volume.shape
    z_window, z_window_detail = select_blind_z_window(volume, blind_z_slices)
    z_start = z_window.start
    z_stop = z_window.stop
    volume = volume[z_window]
    nz, ny, nx = volume.shape
    requested_workers = max_workers
    cpu_workers = resolve_worker_count(requested_workers)
    matlab_threads = min(2, max(1, matlab_threads))
    matlab_workers = max(1, matlab_workers)
    matlab_timeout = max(0, matlab_timeout)
    pad_z = max(0, pad_z)
    snr_weight_cap = max(0.0, snr_weight_cap)
    chunk_xy = resolve_chunk_xy(
        chunk_xy,
        volume.shape,
        volume.dtype,
        overlap_xy=pad_xy,
        vram_gb=vram_gb,
        workers=cpu_workers,
        min_xy=max(128, psf_seed.shape[-1]),
        max_xy=min(DEFAULT_BLIND_CHUNK_XY, ny, nx),
    )
    max_workers, worker_detail = resolve_blind_worker_count(
        requested_workers,
        cpu_workers,
        volume.shape,
        volume.dtype,
        chunk_xy,
        pad_xy,
    )

    print(
        f"  Volume shape: {original_shape}; blind_volume_shape={volume.shape}; "
        f"{z_window_detail}; resolved_chunk_xy={chunk_xy}",
        flush=True,
    )
    matlab_workers = min(matlab_workers, max_workers)
    print(
        f"  Blind worker selection: io_workers={max_workers} ({worker_detail}); "
        f"matlab_workers={matlab_workers}",
        flush=True,
    )

    cache_path = None
    if use_cache:
        cache_root = _resolve_psf_cache_root(image_path, cache_dir)
        cache_key = _psf_cache_key(
            image_path,
            psf_seed,
            n_iters,
            chunk_xy,
            pad_xy,
            pad_z,
            script_dir,
            "snr_weighted_mean",
            snr_weight_cap,
            (z_start, z_stop),
        )
        cache_path = cache_root / f"estimated_psf_{cache_key}.tif"
        if cache_path.exists():
            print(f"Using cached PSF estimate: {cache_path}", flush=True)
            return _normalise_psf(imread(str(cache_path)))

    tile_origins = _tile_origins(ny, nx, chunk_xy)

    print(f"  Processing {len(tile_origins)} chunk(s) of size "
          f"(nz={nz}, xy<={chunk_xy}, halo_xy={pad_xy}, pad_z={pad_z}, "
          f"io_workers={max_workers}, "
          f"matlab_workers={matlab_workers}, "
          f"matlab_threads={matlab_threads}, matlab_timeout={matlab_timeout}s, "
          f"snr_weight_cap={snr_weight_cap:g})...",
          flush=True)

    psf_estimates: list[np.ndarray] = []
    psf_weights: list[float] = []
    failed_chunks = 0
    completed_chunks = 0
    prefetch_chunks = prefetch_chunks if prefetch_chunks > 0 else max_workers
    heartbeat_seconds = 60.0
    last_heartbeat = time.perf_counter()

    with tempfile.TemporaryDirectory(prefix="psf_est_") as tmpdir:
        tmpdir = Path(tmpdir)
        matlab_slots = threading.Semaphore(matlab_workers)
        next_idx = 0
        pending: set[futures.Future] = set()

        with futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            while next_idx < len(tile_origins) or pending:
                submitted_before = next_idx
                while next_idx < len(tile_origins) and len(pending) < prefetch_chunks:
                    pending.add(
                        executor.submit(
                            _estimate_one_tile,
                            next_idx,
                            len(tile_origins),
                            volume,
                            tile_origins[next_idx],
                            psf_seed,
                            pad_xy,
                            pad_z,
                            n_iters,
                            script_dir,
                            tmpdir,
                            matlab_slots,
                            matlab_threads,
                            matlab_timeout,
                            snr_weight_cap,
                        )
                    )
                    next_idx += 1
                if next_idx > submitted_before:
                    print(
                        f"  Submitted blind chunks {submitted_before + 1}-{next_idx}/"
                        f"{len(tile_origins)}; pending={len(pending)}, "
                        f"completed={completed_chunks}, failed={failed_chunks}",
                        flush=True,
                    )

                done, pending = futures.wait(
                    pending,
                    timeout=heartbeat_seconds,
                    return_when=futures.FIRST_COMPLETED,
                )
                if not done:
                    now = time.perf_counter()
                    if now - last_heartbeat >= heartbeat_seconds:
                        print(
                            f"  Blind PSF heartbeat: submitted={next_idx}/"
                            f"{len(tile_origins)}, completed={completed_chunks}, "
                            f"failed={failed_chunks}, pending={len(pending)}",
                            flush=True,
                        )
                        last_heartbeat = now
                    continue

                for future in done:
                    idx, psf_chunk, weight, error = future.result()
                    completed_chunks += 1
                    if error:
                        failed_chunks += 1
                        if "initial PSF must have at least one non-zero element" in error:
                            raise RuntimeError(
                                "MATLAB read the PSF seed as all zeros. "
                                "The seed TIFF writer is incompatible with MATLAB."
                            )
                        if failed_chunks >= 3 and not psf_estimates:
                            for pending_future in pending:
                                pending_future.cancel()
                            raise RuntimeError(
                                "First three chunks failed during PSF estimation; "
                                "aborting instead of submitting every tile to MATLAB."
                            )
                        print(f"  WARNING: chunk {idx} failed, skipping. {error}", flush=True)
                        continue
                    if psf_chunk is not None:
                        psf_estimates.append(psf_chunk)
                        psf_weights.append(weight)

    if not psf_estimates:
        raise RuntimeError(
            "All chunks failed during PSF estimation. "
            "Check MATLAB logs above and ensure deconvblind is available."
        )

    print(f"Merging {len(psf_estimates)} PSF estimate(s) via SNR-weighted mean...", flush=True)
    stack = np.stack(psf_estimates, axis=0)
    weights = np.asarray(psf_weights, dtype=np.float32)
    max_weight = snr_weight_cap if snr_weight_cap > 0 else None
    weights = np.clip(weights, 1e-3, max_weight)
    weights = weights / weights.sum()
    merged = np.tensordot(weights, stack, axes=(0, 0)).astype(np.float32)
    merged = _normalise_psf(merged)

    if cache_path is not None:
        imwrite(str(cache_path), merged)
        print(f"Cached PSF estimate: {cache_path}", flush=True)

    return merged


# ---------------------------------------------------------------------------
# CLI (for standalone testing)
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Estimate PSF from a deskewed TIFF using chunked deconvblind."
    )
    parser.add_argument("--image_path",  required=True)
    parser.add_argument("--output_path", required=True,
                        help="Where to save the merged PSF TIFF.")
    parser.add_argument("--n_iters",    type=int,   default=10)
    parser.add_argument("--chunk_xy",   type=int,   default=DEFAULT_BLIND_CHUNK_XY,
                        help="XY tile size. <=0 auto-sizes from available VRAM.")
    parser.add_argument("--pad_xy",     type=int,   default=32,
                        help="XY halo per edge before deconvblind (pixels).")
    parser.add_argument("--pad_z",      type=int,   default=20,
                        help="Z halo per edge before deconvblind (pixels).")
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
    parser.add_argument("--vram_gb", type=float, default=None,
                        help="Override detected free VRAM in GiB for auto chunk sizing.")
    parser.add_argument("--cache_dir", default=None,
                        help="Directory for cached PSF estimates.")
    parser.add_argument("--no_psf_cache", action="store_true",
                        help="Disable reuse of cached blind PSF estimates.")
    parser.add_argument("--script_dir", default=str(Path(__file__).parent))

    # Optional optical parameters for the PSF seed
    parser.add_argument("--na",         type=float, default=1.0)
    parser.add_argument("--detection_na", type=float, default=None)
    parser.add_argument("--illumination_na", type=float, default=None)
    parser.add_argument("--wavelength", type=float, default=0.525)
    parser.add_argument("--ni",         type=float, default=1.33)
    parser.add_argument("--ns",         type=float, default=None)
    parser.add_argument("--ni0",        type=float, default=None)
    parser.add_argument("--tg",         type=float, default=None)
    parser.add_argument("--tg0",        type=float, default=None)
    parser.add_argument("--ng",         type=float, default=None)
    parser.add_argument("--ng0",        type=float, default=None)
    parser.add_argument("--ti0",        type=float, default=None)
    parser.add_argument("--oversample_factor", type=int, default=3)
    parser.add_argument("--psf_model", choices=("vectorial", "scalar", "gaussian"), default="vectorial")
    parser.add_argument("--camera_pixel_size", type=float, default=None)
    parser.add_argument("--magnification", type=float, default=None)
    parser.add_argument("--dxy",        type=float, default=0.1)
    parser.add_argument("--dz",         type=float, default=0.3)
    parser.add_argument("--psf_size_z", type=int,   default=61)
    parser.add_argument("--psf_size_xy",type=int,   default=128)
    parser.add_argument("--background", type=float, default=0.0)
    args = parser.parse_args()

    dxy = resolve_dxy(args.dxy, args.camera_pixel_size, args.magnification)
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

    merged_psf = estimate_psf_from_chunks(
        image_path=args.image_path,
        psf_seed=psf_seed,
        n_iters=args.n_iters,
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

    imwrite(args.output_path, merged_psf)
    print(f"Merged PSF saved to {args.output_path}", flush=True)


if __name__ == "__main__":
    main()
