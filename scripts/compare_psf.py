"""Compare chunked pipeline deconvolution against the reference MATLAB flow.

The script runs both sides on the same cropped deskewed input volume:

1. Pipeline path: chunked blind PSF estimation followed by CUDA Richardson-Lucy.
2. Reference path: full-window MATLAB deconvblind followed by MATLAB deconvlucy.

It then writes the deconvolved volumes, PSFs, cross-section montages, similarity
metrics, and per-axis Gaussian/FWHM profile summaries for the PSFs.
"""

import argparse
import html
import inspect
import json
import re
import os
import subprocess
import tempfile
from pathlib import Path

import numpy as np
from scipy.optimize import curve_fit
from tifffile import imread, imwrite

# ANTsPy is useful for additional image-similarity metrics, but the comparison
# should still run in older environments that have not rebuilt environment.yml.
try:
    import ants
except Exception as exc:
    ants = None
    ANTS_IMPORT_ERROR = str(exc)
else:
    ANTS_IMPORT_ERROR = ""

from decon_wrapper import deconvolve_tiff
from psf_estimation import (
    _normalise_psf,
    _write_chunk,
    _write_matlab_stack,
    generate_theoretical_psf,
    estimate_psf_from_chunks,
    open_tiff_memmap,
    resolve_dxy,
    select_blind_z_window,
)

CHANNEL_TIMEPOINT_RE = re.compile(r"^CH(?P<channel>\d+)_(?P<timepoint>\d+)(?:_registered_consistent)?$")
GAUSSIAN_FWHM_FACTOR = 2.0 * np.sqrt(2.0 * np.log(2.0))
SSIM_K1 = 0.01
SSIM_K2 = 0.03


def _parse_int_filter(value: str | None) -> set[int] | None:
    """Parse Nextflow-style comma/space integer filters into a set."""
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    tokens = [token for token in re.split(r"[\s,]+", text) if token]
    return {int(token) for token in tokens}


def _find_input_tiff(
    image_dir: Path,
    index: int,
    channels: set[int] | None,
    timepoints: set[int] | None,
) -> Path:
    """Select one CH/timepoint TIFF from a deskewed image directory."""
    tiffs = sorted(
        list(image_dir.glob("CH*.tif"))
        + list(image_dir.glob("CH*.tiff"))
    )
    filtered = []
    for tiff in tiffs:
        match = CHANNEL_TIMEPOINT_RE.match(tiff.stem)
        if not match:
            continue
        channel = int(match.group("channel"))
        timepoint = int(match.group("timepoint"))
        if channels is not None and channel not in channels:
            continue
        if timepoints is not None and timepoint not in timepoints:
            continue
        filtered.append(tiff)
    tiffs = filtered
    if not tiffs:
        raise FileNotFoundError(f"No matching CH*.tif TIFFs found in {image_dir}")
    if index < 0 or index >= len(tiffs):
        raise IndexError(f"TIFF index {index} out of range for {len(tiffs)} file(s)")
    return tiffs[index]


def _center_crop_xy(volume: np.ndarray, crop_xy: int) -> np.ndarray:
    """Return a centered XY crop while preserving all selected Z planes."""
    if crop_xy <= 0:
        return np.asarray(volume)
    _, ny, nx = volume.shape
    crop_y = min(crop_xy, ny)
    crop_x = min(crop_xy, nx)
    y0 = max(0, (ny - crop_y) // 2)
    x0 = max(0, (nx - crop_x) // 2)
    return np.asarray(volume[:, y0:y0 + crop_y, x0:x0 + crop_x])


def _resolve_comparison_crop_xy(volume_shape: tuple[int, int, int], requested_xy: int, chunk_xy: int) -> int:
    """Choose an XY comparison window large enough to exercise multiple chunks."""
    if requested_xy <= 0:
        return requested_xy
    _, ny, nx = volume_shape
    available_xy = min(ny, nx)
    min_multi_chunk_xy = max(1, chunk_xy) * 2
    resolved = max(requested_xy, min_multi_chunk_xy)
    return min(resolved, available_xy)


def _resolve_comparison_z_slices(
    volume_shape: tuple[int, int, int],
    requested_z: int,
    psf_size_z: int,
    pad_z: int,
) -> int:
    """Choose a Z window that covers requested slices, PSF depth, and padding."""
    available_z = volume_shape[0]
    if requested_z <= 0:
        return requested_z
    min_z = max(requested_z, psf_size_z, max(1, pad_z) * 4)
    return min(min_z, available_z)


def _run_full_blind(
    volume: np.ndarray,
    psf_seed: np.ndarray,
    n_iters: int,
    pad_xy: int,
    pad_z: int,
    script_dir: Path,
    matlab_threads: int,
    matlab_timeout: int,
) -> np.ndarray:
    """Run the reference first pass on the full comparison crop in MATLAB."""
    with tempfile.TemporaryDirectory(prefix="full_blind_psf_") as tmpdir:
        tmpdir = Path(tmpdir)
        chunk_path = tmpdir / "full_window.tif"
        seed_path = tmpdir / "seed.tif"
        psf_out_path = tmpdir / "full_blind_psf.tif"
        _write_chunk(volume, chunk_path)
        _write_matlab_stack(psf_seed, seed_path, scale_float=True)

        # Keep MATLAB thread counts bounded so Slurm CPU requests stay honest.
        matlab_threads = min(2, max(1, matlab_threads))
        pad_xy = max(0, pad_xy)
        pad_z = max(0, pad_z)
        matlab_thread_cmd = f"maxNumCompThreads({matlab_threads}); "
        pad_cmd = (
            f"chunk = padarray(chunk, [{pad_xy} {pad_xy} {pad_z}], 'symmetric'); "
            if pad_xy > 0 or pad_z > 0 else ""
        )
        # This mirrors the reference script's deconvblind step but limits input
        # to the selected comparison crop instead of the whole experimental TIFF.
        matlab_cmd = (
            f"addpath('{script_dir}'); "
            f"{matlab_thread_cmd}"
            f"chunk = single(readtiffstack('{chunk_path}')); "
            f"psf_seed = single(readtiffstack('{seed_path}')); "
            f"psf_seed = psf_seed / sum(psf_seed(:)); "
            f"{pad_cmd}"
            f"[~, psf_est] = deconvblind(chunk, psf_seed, {n_iters}); "
            f"psf_est = single(psf_est); "
            f"psf_est = psf_est / sum(psf_est(:)); "
            f"writetiffstack(psf_est, '{psf_out_path}');"
        )
        matlab_args = ["matlab"]
        if matlab_threads == 1:
            matlab_args.append("-singleCompThread")
        matlab_args.extend(["-batch", matlab_cmd])
        env = os.environ.copy()
        # MATLAB and BLAS libraries can otherwise oversubscribe CPU threads.
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
                f"MATLAB full-window deconvblind timed out after {matlab_timeout}s.\n"
                f"STDOUT: {exc.stdout or ''}\nSTDERR: {exc.stderr or ''}"
            ) from exc
        if result.returncode != 0:
            raise RuntimeError(
                f"MATLAB full-window deconvblind failed (returncode={result.returncode}).\n"
                f"STDOUT: {result.stdout}\nSTDERR: {result.stderr}"
            )
        if not psf_out_path.exists():
            raise RuntimeError("MATLAB produced no full-blind PSF output")
        psf = imread(str(psf_out_path)).astype(np.float32)
    if psf.shape != psf_seed.shape:
        raise ValueError(f"Full-blind PSF shape {psf.shape} != seed shape {psf_seed.shape}")
    return _normalise_psf(psf)


def _run_matlab_deconvlucy(
    volume: np.ndarray,
    psf: np.ndarray,
    n_iters: int,
    pad_xy: int,
    pad_z: int,
    script_dir: Path,
    matlab_threads: int,
    matlab_timeout: int,
) -> np.ndarray:
    """
    Mirror the reference script's second pass:
    pad image -> deconvlucy(image, psfr, iter) -> crop -> rescale to input range.
    """
    with tempfile.TemporaryDirectory(prefix="full_lucy_decon_") as tmpdir:
        tmpdir = Path(tmpdir)
        image_path = tmpdir / "full_window.tif"
        psf_path = tmpdir / "psfr.tif"
        output_path = tmpdir / "Dec2.tif"
        _write_chunk(volume, image_path)
        _write_matlab_stack(psf, psf_path, scale_float=True)

        matlab_threads = min(2, max(1, matlab_threads))
        pad_xy = max(0, pad_xy)
        pad_z = max(0, pad_z)
        matlab_thread_cmd = f"maxNumCompThreads({matlab_threads}); "
        pad_cmd = (
            f"E1 = padarray(E1, [{pad_xy} {pad_xy} {pad_z}], 'symmetric'); "
            if pad_xy > 0 or pad_z > 0 else ""
        )
        crop_cmd = (
            f"Dec2 = Dec2({pad_xy + 1}:{pad_xy}+mImage, "
            f"{pad_xy + 1}:{pad_xy}+nImage, "
            f"{pad_z + 1}:{pad_z}+NumberImages); "
        )
        # Use MATLAB for this path so the reference Dec2 output matches the
        # original script's Lucy-Richardson implementation and normalization.
        matlab_cmd = (
            f"addpath('{script_dir}'); "
            f"{matlab_thread_cmd}"
            f"FinalImage = readtiffstack('{image_path}'); "
            f"mImage = size(FinalImage, 1); "
            f"nImage = size(FinalImage, 2); "
            f"NumberImages = size(FinalImage, 3); "
            f"E1 = single(FinalImage); "
            f"maxE1 = max(E1(:)); "
            f"minE1 = min(E1(:)); "
            f"psfr = single(readtiffstack('{psf_path}')); "
            f"psfr = psfr / sum(psfr(:)); "
            f"{pad_cmd}"
            f"Dec2 = deconvlucy(E1, psfr, {n_iters}); "
            f"{crop_cmd}"
            f"decMin = min(Dec2(:)); "
            f"decMax = max(Dec2(:)); "
            f"if decMax > decMin; "
            f"Dec2 = (Dec2 - decMin) / (decMax - decMin); "
            f"Dec2 = times(Dec2, maxE1 - minE1) + minE1; "
            f"end; "
            f"Dec2 = uint16(max(0, min(65535, Dec2))); "
            f"writetiffstack(Dec2, '{output_path}');"
        )

        matlab_args = ["matlab"]
        if matlab_threads == 1:
            matlab_args.append("-singleCompThread")
        matlab_args.extend(["-batch", matlab_cmd])
        env = os.environ.copy()
        # Clamp numerical library threads consistently with maxNumCompThreads.
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
                f"MATLAB deconvlucy timed out after {matlab_timeout}s.\n"
                f"STDOUT: {exc.stdout or ''}\nSTDERR: {exc.stderr or ''}"
            ) from exc
        if result.returncode != 0:
            raise RuntimeError(
                f"MATLAB deconvlucy failed (returncode={result.returncode}).\n"
                f"STDOUT: {result.stdout}\nSTDERR: {result.stderr}"
            )
        if not output_path.exists():
            raise RuntimeError("MATLAB produced no Lucy-Richardson Dec2 output")
        return imread(str(output_path)).astype(np.uint16, copy=False)


def _fwhm_pixels(line: np.ndarray) -> float:
    """Estimate half-max FWHM in pixels with linear edge interpolation."""
    line = np.asarray(line, dtype=np.float32)
    if line.size == 0 or float(line.max()) <= 0:
        return 0.0
    half = float(line.max()) * 0.5
    above = np.flatnonzero(line >= half)
    if above.size == 0:
        return 0.0
    left = float(above[0])
    right = float(above[-1])
    if above[0] > 0:
        i = above[0]
        denom = float(line[i] - line[i - 1])
        if denom != 0:
            left = (i - 1) + (half - float(line[i - 1])) / denom
    if above[-1] < line.size - 1:
        i = above[-1]
        denom = float(line[i + 1] - line[i])
        if denom != 0:
            right = i + (half - float(line[i])) / denom
    return max(0.0, right - left)


def _gaussian_1d(x: np.ndarray, baseline: float, amplitude: float, center: float, sigma: float) -> np.ndarray:
    """One-dimensional Gaussian model with a constant baseline."""
    return baseline + amplitude * np.exp(-0.5 * ((x - center) / sigma) ** 2)


def _fit_gaussian_profile(
    line: np.ndarray,
    spacing_um: float,
    axis_label: str,
    peak_index: int,
) -> dict:
    """Fit one PSF axis profile and return raw values plus fitted parameters."""
    values = np.asarray(line, dtype=np.float64)
    x_px = np.arange(values.size, dtype=np.float64)
    baseline0 = float(np.percentile(values, 5))
    amplitude0 = max(float(values.max() - baseline0), np.finfo(float).eps)
    sigma0 = max(_fwhm_pixels(values) / GAUSSIAN_FWHM_FACTOR, 1.0)
    center0 = float(peak_index)
    # Bounds keep the optimizer on a positive-width, in-profile peak while still
    # allowing a nonzero baseline for imperfect or noisy blind PSFs.
    bounds = (
        [float(values.min()) - abs(amplitude0), 0.0, 0.0, np.finfo(float).eps],
        [float(values.max()), float(values.max() - values.min()) * 4.0 + np.finfo(float).eps, float(values.size - 1), float(values.size)],
    )
    try:
        params, covariance = curve_fit(
            _gaussian_1d,
            x_px,
            values,
            p0=[baseline0, amplitude0, center0, sigma0],
            bounds=bounds,
            maxfev=20000,
        )
        fitted = _gaussian_1d(x_px, *params)
        residual = values - fitted
        ss_res = float(np.sum(residual * residual))
        ss_tot = float(np.sum((values - values.mean()) ** 2))
        r_squared = 1.0 - ss_res / ss_tot if ss_tot > 0 else 1.0
        sigma_px = abs(float(params[3]))
        center_px = float(params[2])
        fwhm_px = GAUSSIAN_FWHM_FACTOR * sigma_px
        fit_status = "ok"
        error = ""
        covariance_diag = [float(v) for v in np.diag(covariance)]
    except Exception as exc:
        fitted = np.full_like(values, np.nan, dtype=np.float64)
        sigma_px = float("nan")
        center_px = float("nan")
        fwhm_px = float("nan")
        r_squared = float("nan")
        fit_status = "failed"
        error = str(exc)
        covariance_diag = []
        params = [float("nan"), float("nan"), float("nan"), float("nan")]

    return {
        "axis": axis_label,
        "spacing_um": float(spacing_um),
        "peak_index_px": int(peak_index),
        "half_max_fwhm_px": _fwhm_pixels(values),
        "half_max_fwhm_um": _fwhm_pixels(values) * spacing_um,
        "gaussian_baseline": float(params[0]),
        "gaussian_amplitude": float(params[1]),
        "gaussian_center_px": center_px,
        "gaussian_sigma_px": sigma_px,
        "gaussian_fwhm_px": fwhm_px,
        "gaussian_fwhm_um": fwhm_px * spacing_um,
        "gaussian_r_squared": r_squared,
        "fit_status": fit_status,
        "fit_error": error,
        "covariance_diag": covariance_diag,
        "profile": [float(v) for v in values],
        "fit_profile": [float(v) for v in fitted],
    }


def _psf_stats(name: str, psf: np.ndarray, dxy: float, dz: float) -> dict:
    """Measure PSF peak location and X/Y/Z FWHM through its brightest voxel."""
    zc, yc, xc = np.unravel_index(int(np.argmax(psf)), psf.shape)
    profiles = {
        "x": _fit_gaussian_profile(psf[zc, yc, :], dxy, "x", xc),
        "y": _fit_gaussian_profile(psf[zc, :, xc], dxy, "y", yc),
        "z": _fit_gaussian_profile(psf[:, yc, xc], dz, "z", zc),
    }
    fwhm_x = profiles["x"]["half_max_fwhm_px"]
    fwhm_y = profiles["y"]["half_max_fwhm_px"]
    fwhm_z = profiles["z"]["half_max_fwhm_px"]
    return {
        "name": name,
        "shape": list(psf.shape),
        "peak_index_zyx": [int(zc), int(yc), int(xc)],
        "sum": float(psf.sum()),
        "max": float(psf.max()),
        "fwhm_x_px": fwhm_x,
        "fwhm_y_px": fwhm_y,
        "fwhm_z_px": fwhm_z,
        "fwhm_x_um": fwhm_x * dxy,
        "fwhm_y_um": fwhm_y * dxy,
        "fwhm_z_um": fwhm_z * dz,
        "axis_profiles": profiles,
    }


def _ants_similarity_metrics(a: np.ndarray, b: np.ndarray) -> dict:
    """Compute optional ANTsPy image similarity metrics for aligned volumes."""
    result = {
        "available": ants is not None,
        "package": "antspyx",
        "import_name": "ants",
        "import_error": ANTS_IMPORT_ERROR,
        "metrics": {},
    }
    if ants is None:
        return result

    fixed = ants.from_numpy(np.asarray(a, dtype=np.float32))
    moving = ants.from_numpy(np.asarray(b, dtype=np.float32))
    for metric_name in ("Correlation", "MeanSquares", "ANTSNeighborhoodCorrelation"):
        try:
            result["metrics"][metric_name] = float(
                ants.image_similarity(fixed, moving, metric_type=metric_name)
            )
        except Exception as exc:
            result["metrics"][metric_name] = None
            result[f"{metric_name}_error"] = str(exc)
    return result


def _pair_metrics(a: np.ndarray, b: np.ndarray) -> dict:
    """Compute scalar similarity and error metrics for two aligned volumes."""
    av = a.ravel().astype(np.float64)
    bv = b.ravel().astype(np.float64)
    da = av - av.mean()
    db = bv - bv.mean()
    denom = np.linalg.norm(da) * np.linalg.norm(db)
    pearson = float(np.dot(da, db) / denom) if denom > 0 else 0.0

    # This is a global SSIM over the full aligned crop. It is not windowed SSIM,
    # so it is intended as a single coarse agreement score.
    data_range = max(float(av.max()), float(bv.max())) - min(float(av.min()), float(bv.min()))
    if data_range <= 0:
        ssim = 1.0
        c1 = 0.0
        c2 = 0.0
    else:
        c1 = (SSIM_K1 * data_range) ** 2
        c2 = (SSIM_K2 * data_range) ** 2
        mux = float(av.mean())
        muy = float(bv.mean())
        varx = float(av.var())
        vary = float(bv.var())
        cov = float(((av - mux) * (bv - muy)).mean())
        ssim = ((2 * mux * muy + c1) * (2 * cov + c2)) / (
            (mux * mux + muy * muy + c1) * (varx + vary + c2)
        )

    diff = av - bv
    return {
        "ncc": pearson,
        "pearson_correlation": pearson,
        "correlation_coefficient": pearson,
        "correlation_method": "Pearson correlation of flattened aligned volumes",
        "antspy_image_similarity": _ants_similarity_metrics(a, b),
        "ssim_global": float(ssim),
        "ssim_method": "Global SSIM formula over flattened aligned volumes",
        "ssim_k1": SSIM_K1,
        "ssim_k2": SSIM_K2,
        "ssim_c1": float(c1),
        "ssim_c2": float(c2),
        "ssim_data_range": float(data_range),
        "mse": float(np.mean(diff * diff)),
        "mae": float(np.mean(np.abs(diff))),
        "max_abs_diff": float(np.max(np.abs(diff))),
    }


def _volume_stats(name: str, volume: np.ndarray) -> dict:
    """Summarize intensity distribution and shape for an output volume."""
    values = np.asarray(volume).astype(np.float64, copy=False)
    return {
        "name": name,
        "shape": list(volume.shape),
        "dtype": str(volume.dtype),
        "min": float(np.min(values)),
        "max": float(np.max(values)),
        "mean": float(np.mean(values)),
        "std": float(np.std(values)),
        "p1": float(np.percentile(values, 1)),
        "p50": float(np.percentile(values, 50)),
        "p99": float(np.percentile(values, 99)),
    }


def _center_crop_to_shape(volume: np.ndarray, shape: tuple[int, int, int]) -> np.ndarray:
    """Center-crop a 3-D volume to an exact target shape."""
    if volume.ndim != 3:
        raise ValueError(f"Expected 3-D volume, got shape {volume.shape}")
    slices = []
    for current, target in zip(volume.shape, shape):
        if target > current:
            raise ValueError(f"Cannot crop shape {volume.shape} to larger shape {shape}")
        start = (current - target) // 2
        slices.append(slice(start, start + target))
    return np.asarray(volume[tuple(slices)])


def _align_decon_outputs(
    pipeline: np.ndarray,
    reference: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, dict]:
    """Align pipeline/reference outputs by center-cropping to common shape."""
    if pipeline.ndim != 3 or reference.ndim != 3:
        raise ValueError(
            f"Expected 3-D deconvolution outputs, got "
            f"pipeline={pipeline.shape}, reference={reference.shape}"
        )
    common_shape = tuple(min(a, b) for a, b in zip(pipeline.shape, reference.shape))
    alignment = {
        "original_pipeline_shape_zyx": list(pipeline.shape),
        "original_reference_shape_zyx": list(reference.shape),
        "comparison_shape_zyx": list(common_shape),
        "center_cropped_for_comparison": pipeline.shape != reference.shape,
    }
    return (
        _center_crop_to_shape(pipeline, common_shape),
        _center_crop_to_shape(reference, common_shape),
        alignment,
    )


def _normalise_plane(plane: np.ndarray) -> np.ndarray:
    """Scale a 2-D plane to uint16 for montage visualization."""
    plane = np.asarray(plane, dtype=np.float32)
    plane = plane - float(plane.min())
    max_value = float(plane.max())
    if max_value > 0:
        plane = plane / max_value
    return np.clip(np.rint(plane * 65535), 0, 65535).astype(np.uint16)


def _resize_nearest(plane: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    """Resize a 2-D plane with nearest-neighbor sampling for quick montages."""
    out_y, out_x = shape
    in_y, in_x = plane.shape
    y_idx = np.linspace(0, in_y - 1, out_y).astype(int)
    x_idx = np.linspace(0, in_x - 1, out_x).astype(int)
    return plane[np.ix_(y_idx, x_idx)]


def _make_cross_section_montage(psfs: dict[str, np.ndarray]) -> np.ndarray:
    """Build a TIFF stack of PSF XY/XZ/YZ views and difference views."""
    panels = []
    target_shape = (160, 160)
    for name in ("theoretical", "chunked_blind", "full_blind"):
        psf = psfs[name]
        zc, yc, xc = np.unravel_index(int(np.argmax(psf)), psf.shape)
        planes = [
            psf[zc, :, :],
            psf[:, yc, :],
            psf[:, :, xc],
        ]
        row = [_resize_nearest(_normalise_plane(p), target_shape) for p in planes]
        panels.append(np.concatenate(row, axis=1))
    for left, right in (("chunked_blind", "full_blind"), ("chunked_blind", "theoretical")):
        diff = np.abs(psfs[left] - psfs[right])
        zc, yc, xc = np.unravel_index(int(np.argmax(psfs[left])), psfs[left].shape)
        planes = [
            diff[zc, :, :],
            diff[:, yc, :],
            diff[:, :, xc],
        ]
        row = [_resize_nearest(_normalise_plane(p), target_shape) for p in planes]
        panels.append(np.concatenate(row, axis=1))
    return np.stack(panels, axis=0)


def _make_decon_montage(volumes: dict[str, np.ndarray]) -> np.ndarray:
    """Build a TIFF stack showing pipeline, reference, and absolute difference."""
    panels = []
    target_shape = (256, 256)
    reference = volumes["reference_matlab_dec2"]
    zc = reference.shape[0] // 2
    yc = reference.shape[1] // 2
    xc = reference.shape[2] // 2
    for name in ("pipeline_cuda_db2", "reference_matlab_dec2"):
        volume = volumes[name]
        planes = [
            volume[zc, :, :],
            volume[:, yc, :],
            volume[:, :, xc],
        ]
        row = [_resize_nearest(_normalise_plane(p), target_shape) for p in planes]
        panels.append(np.concatenate(row, axis=1))

    diff = np.abs(
        volumes["pipeline_cuda_db2"].astype(np.float32)
        - volumes["reference_matlab_dec2"].astype(np.float32)
    )
    planes = [
        diff[zc, :, :],
        diff[:, yc, :],
        diff[:, :, xc],
    ]
    row = [_resize_nearest(_normalise_plane(p), target_shape) for p in planes]
    panels.append(np.concatenate(row, axis=1))
    return np.stack(panels, axis=0)


def _write_metrics(metrics: dict, output_dir: Path) -> None:
    """Write JSON plus compact TSVs for decon metrics and PSF Gaussian fits."""
    (output_dir / "decon_metrics.json").write_text(
        json.dumps(metrics, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    rows = [
        "comparison\tpearson_correlation\tncc\tssim_global\tssim_data_range\t"
        "ssim_k1\tssim_k2\tantspy_available\tantspy_correlation\t"
        "antspy_mean_squares\tantspy_neighborhood_correlation\tmse\tmae\tmax_abs_diff"
    ]
    for name, values in metrics["comparisons"].items():
        ants_metrics = values["antspy_image_similarity"]["metrics"]
        rows.append(
            "\t".join(
                [
                    name,
                    f"{values['pearson_correlation']:.8g}",
                    f"{values['ncc']:.8g}",
                    f"{values['ssim_global']:.8g}",
                    f"{values['ssim_data_range']:.8g}",
                    f"{values['ssim_k1']:.8g}",
                    f"{values['ssim_k2']:.8g}",
                    str(values["antspy_image_similarity"]["available"]),
                    "" if ants_metrics.get("Correlation") is None else f"{ants_metrics['Correlation']:.8g}",
                    "" if ants_metrics.get("MeanSquares") is None else f"{ants_metrics['MeanSquares']:.8g}",
                    "" if ants_metrics.get("ANTSNeighborhoodCorrelation") is None else f"{ants_metrics['ANTSNeighborhoodCorrelation']:.8g}",
                    f"{values['mse']:.8g}",
                    f"{values['mae']:.8g}",
                    f"{values['max_abs_diff']:.8g}",
                ]
            )
        )
    (output_dir / "decon_metrics.tsv").write_text("\n".join(rows) + "\n", encoding="utf-8")

    fwhm_rows = [
        "psf\taxis\tspacing_um\tpeak_index_px\thalf_max_fwhm_px\thalf_max_fwhm_um\t"
        "gaussian_center_px\tgaussian_sigma_px\tgaussian_fwhm_px\tgaussian_fwhm_um\t"
        "gaussian_r_squared\tfit_status\tfit_error"
    ]
    profile_rows = [
        "psf\taxis\tcoordinate_px\tcoordinate_relative_px\tcoordinate_um\tcoordinate_relative_um\t"
        "intensity\tgaussian_fit_intensity"
    ]
    for psf_name, psf_values in metrics["psfs"].items():
        for axis, profile in psf_values["axis_profiles"].items():
            fwhm_rows.append(
                "\t".join(
                    [
                        psf_name,
                        axis,
                        f"{profile['spacing_um']:.8g}",
                        str(profile["peak_index_px"]),
                        f"{profile['half_max_fwhm_px']:.8g}",
                        f"{profile['half_max_fwhm_um']:.8g}",
                        f"{profile['gaussian_center_px']:.8g}",
                        f"{profile['gaussian_sigma_px']:.8g}",
                        f"{profile['gaussian_fwhm_px']:.8g}",
                        f"{profile['gaussian_fwhm_um']:.8g}",
                        f"{profile['gaussian_r_squared']:.8g}",
                        profile["fit_status"],
                        profile["fit_error"].replace("\t", " ").replace("\n", " "),
                    ]
                )
            )
            fit_values = profile["fit_profile"]
            for index, intensity in enumerate(profile["profile"]):
                profile_rows.append(
                    "\t".join(
                        [
                            psf_name,
                            axis,
                            str(index),
                            str(index - profile["peak_index_px"]),
                            f"{index * profile['spacing_um']:.8g}",
                            f"{(index - profile['peak_index_px']) * profile['spacing_um']:.8g}",
                            f"{intensity:.8g}",
                            f"{fit_values[index]:.8g}",
                        ]
                    )
                )
    (output_dir / "psf_fwhm.tsv").write_text("\n".join(fwhm_rows) + "\n", encoding="utf-8")
    (output_dir / "psf_axis_profiles.tsv").write_text(
        "\n".join(profile_rows) + "\n",
        encoding="utf-8",
    )


def _polyline_points(xs: np.ndarray, ys: np.ndarray, x0: float, y0: float, width: float, height: float) -> str:
    """Convert profile coordinates into SVG polyline point text."""
    finite = np.isfinite(xs) & np.isfinite(ys)
    if not np.any(finite):
        return ""
    xs = xs[finite]
    ys = ys[finite]
    xmin = float(xs.min())
    xmax = float(xs.max())
    ymin = float(ys.min())
    ymax = float(ys.max())
    if xmax <= xmin:
        xmax = xmin + 1.0
    if ymax <= ymin:
        ymax = ymin + 1.0
    px = x0 + (xs - xmin) / (xmax - xmin) * width
    py = y0 + height - (ys - ymin) / (ymax - ymin) * height
    return " ".join(f"{x:.2f},{y:.2f}" for x, y in zip(px, py))


def _write_psf_profile_svg(metrics: dict, output_dir: Path) -> None:
    """Write a dependency-free SVG preview of raw PSF profiles and fits."""
    psfs = metrics["psfs"]
    psf_names = list(psfs.keys())
    axes = ("x", "y", "z")
    panel_w = 330
    panel_h = 210
    margin_l = 55
    margin_t = 42
    gap_x = 28
    gap_y = 34
    width = margin_l + len(axes) * panel_w + (len(axes) - 1) * gap_x + 30
    height = margin_t + len(psf_names) * panel_h + (len(psf_names) - 1) * gap_y + 35
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        '<style>text{font-family:Arial,sans-serif;font-size:12px}.title{font-size:14px;font-weight:bold}'
        '.small{font-size:10px}</style>',
    ]
    for row, psf_name in enumerate(psf_names):
        for col, axis in enumerate(axes):
            profile = psfs[psf_name]["axis_profiles"][axis]
            x = (
                np.arange(len(profile["profile"]), dtype=np.float64)
                - float(profile["peak_index_px"])
            ) * float(profile["spacing_um"])
            raw = np.asarray(profile["profile"], dtype=np.float64)
            fit = np.asarray(profile["fit_profile"], dtype=np.float64)
            x0 = margin_l + col * (panel_w + gap_x)
            y0 = margin_t + row * (panel_h + gap_y)
            plot_w = panel_w - 38
            plot_h = panel_h - 58
            plot_x = x0 + 25
            plot_y = y0 + 32
            raw_points = _polyline_points(x, raw, plot_x, plot_y, plot_w, plot_h)
            fit_points = _polyline_points(x, fit, plot_x, plot_y, plot_w, plot_h)
            title = (
                f"{psf_name} {axis.upper()}  "
                f"FWHM={profile['gaussian_fwhm_um']:.4g} um  "
                f"R2={profile['gaussian_r_squared']:.4g}"
            )
            parts.extend(
                [
                    f'<text x="{x0}" y="{y0 + 14}" class="title">{html.escape(title)}</text>',
                    f'<rect x="{plot_x}" y="{plot_y}" width="{plot_w}" height="{plot_h}" '
                    'fill="#f8f8f8" stroke="#999"/>',
                    f'<line x1="{plot_x}" y1="{plot_y + plot_h}" x2="{plot_x + plot_w}" '
                    f'y2="{plot_y + plot_h}" stroke="#555"/>',
                    f'<line x1="{plot_x}" y1="{plot_y}" x2="{plot_x}" '
                    f'y2="{plot_y + plot_h}" stroke="#555"/>',
                ]
            )
            if raw_points:
                parts.append(
                    f'<polyline points="{raw_points}" fill="none" stroke="#1f77b4" '
                    'stroke-width="1.6"/>'
                )
            if fit_points:
                parts.append(
                    f'<polyline points="{fit_points}" fill="none" stroke="#d62728" '
                    'stroke-width="1.6"/>'
                )
            xmin = float(x.min()) if x.size else 0.0
            xmax = float(x.max()) if x.size else 0.0
            ymax = float(np.nanmax(raw)) if raw.size else 0.0
            parts.extend(
                [
                    f'<text x="{plot_x}" y="{plot_y + plot_h + 16}" class="small">{xmin:.3g} um</text>',
                    f'<text x="{plot_x + plot_w - 45}" y="{plot_y + plot_h + 16}" class="small">{xmax:.3g} um</text>',
                    f'<text x="{plot_x + 4}" y="{plot_y + 12}" class="small">max {ymax:.3g}</text>',
                    f'<text x="{plot_x + plot_w - 115}" y="{plot_y + 14}" class="small" fill="#1f77b4">raw</text>',
                    f'<text x="{plot_x + plot_w - 72}" y="{plot_y + 14}" class="small" fill="#d62728">Gaussian</text>',
                ]
            )
    parts.append("</svg>")
    (output_dir / "psf_axis_profiles.svg").write_text("\n".join(parts) + "\n", encoding="utf-8")


def main() -> None:
    """Parse CLI arguments, run both deconvolution paths, and write artifacts."""
    parser = argparse.ArgumentParser(
        description=(
            "Compare this pipeline's chunked blind + CUDA RL deconvolution "
            "against the reference MATLAB deconvblind -> deconvlucy output."
        )
    )
    parser.add_argument("--image_path", required=True, help="Deskewed Top_shear directory.")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--script_dir", default=str(Path(__file__).parent))
    parser.add_argument("--channels", default="")
    parser.add_argument("--timepoints", default="")
    parser.add_argument("--tiff_index", type=int, default=0)
    parser.add_argument("--sanity_xy", type=int, default=768,
                        help="Center XY crop for comparison. <=0 uses full XY.")
    parser.add_argument("--blind_iters", type=int, default=3)
    parser.add_argument("--chunk_xy", type=int, default=256)
    parser.add_argument("--blind_passes", type=int, default=2,
                        help="Chunked blind PSF passes for the pipeline-style output.")
    parser.add_argument("--decon_chunk_xy", type=int, default=0,
                        help="Core XY tile size for CUDA deconvolution. <=0 auto-sizes from VRAM.")
    parser.add_argument("--pad_xy", type=int, default=32)
    parser.add_argument("--pad_z", type=int, default=20)
    parser.add_argument("--blind_workers", type=int, default=1)
    parser.add_argument("--matlab_threads", type=int, default=1)
    parser.add_argument("--matlab_workers", type=int, default=1)
    parser.add_argument("--matlab_timeout", type=int, default=1800)
    parser.add_argument("--lucy_iters", type=int, default=10,
                        help="MATLAB deconvlucy iterations for the second-pass Dec2 output.")
    parser.add_argument("--blind_z_slices", type=int, default=128)
    parser.add_argument("--snr_weight_cap", type=float, default=100.0)
    parser.add_argument("--prefetch_chunks", type=int, default=0)
    parser.add_argument("--vram_gb", type=float, default=None)
    parser.add_argument("--decon_workers", type=int, default=1)
    parser.add_argument("--overlap_xy", type=int, default=0,
                        help="Override CUDA decon XY overlap. <=0 uses the pipeline default.")
    parser.add_argument("--na", type=float, default=1.0)
    parser.add_argument("--detection_na", type=float, default=None)
    parser.add_argument("--illumination_na", type=float, default=None)
    parser.add_argument("--wavelength", type=float, default=0.520)
    parser.add_argument("--ni", type=float, default=1.515)
    parser.add_argument("--ns", type=float, default=None)
    parser.add_argument("--ni0", type=float, default=None)
    parser.add_argument("--tg", type=float, default=None)
    parser.add_argument("--tg0", type=float, default=None)
    parser.add_argument("--ng", type=float, default=None)
    parser.add_argument("--ng0", type=float, default=None)
    parser.add_argument("--ti0", type=float, default=None)
    parser.add_argument("--oversample_factor", type=int, default=3)
    parser.add_argument("--psf_model", choices=("vectorial", "scalar", "gaussian"), default="vectorial")
    parser.add_argument("--camera_pixel_size", type=float, default=None)
    parser.add_argument("--magnification", type=float, default=None)
    parser.add_argument("--dxy", type=float, default=0.118)
    parser.add_argument("--dz", type=float, default=0.118)
    parser.add_argument("--psf_size_z", type=int, default=101)
    parser.add_argument("--psf_size_xy", type=int, default=61)
    parser.add_argument("--background", type=float, default=0.0)
    args = parser.parse_args()

    image_dir = Path(args.image_path)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    script_dir = Path(args.script_dir)

    channels = _parse_int_filter(args.channels)
    timepoints = _parse_int_filter(args.timepoints)
    input_tiff = _find_input_tiff(image_dir, args.tiff_index, channels, timepoints)
    print(f"Using TIFF for deconvolution comparison: {input_tiff}", flush=True)

    # Generate the same theoretical PSF seed for both the chunked pipeline path
    # and the full-window MATLAB reference path.
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

    # The comparison crop is intentionally larger than one chunk where possible
    # so the chunked PSF estimate exercises stitching/aggregation behavior.
    volume = open_tiff_memmap(input_tiff)
    comparison_z_slices = _resolve_comparison_z_slices(
        volume.shape,
        args.blind_z_slices,
        args.psf_size_z,
        args.pad_z,
    )
    if comparison_z_slices != args.blind_z_slices:
        print(
            f"Adjusted comparison Z window from {args.blind_z_slices} to "
            f"{comparison_z_slices} slices to cover PSF support and padding where possible.",
            flush=True,
        )
    z_window, z_detail = select_blind_z_window(volume, comparison_z_slices)
    comparison_xy = _resolve_comparison_crop_xy(volume.shape, args.sanity_xy, args.chunk_xy)
    if comparison_xy != args.sanity_xy:
        print(
            f"Adjusted comparison XY crop from {args.sanity_xy} to {comparison_xy} "
            f"to cover multiple chunks where possible.",
            flush=True,
        )
    cropped = _center_crop_xy(volume[z_window], comparison_xy)
    crop_tiff = output_dir / "sanity_input_crop.tif"
    imwrite(str(crop_tiff), cropped)
    print(
        f"Saved sanity input crop: {crop_tiff} shape={cropped.shape}; {z_detail}",
        flush=True,
    )

    print("Estimating pipeline-style chunked blind PSF on comparison crop...", flush=True)
    psf_estimation_kwargs = {
        "image_path": crop_tiff,
        "psf_seed": psf_seed,
        "n_iters": args.blind_iters,
        "chunk_xy": args.chunk_xy,
        "pad_xy": args.pad_xy,
        "pad_z": args.pad_z,
        "script_dir": script_dir,
        "max_workers": args.blind_workers,
        "prefetch_chunks": args.prefetch_chunks,
        "vram_gb": args.vram_gb,
        "cache_dir": output_dir / ".psf_cache",
        "use_cache": False,
        "matlab_threads": args.matlab_threads,
        "matlab_workers": args.matlab_workers,
        "matlab_timeout": args.matlab_timeout,
        "snr_weight_cap": args.snr_weight_cap,
        "blind_z_slices": comparison_z_slices,
    }
    # Keep compatibility with older psf_estimation.py versions that predate
    # blind_passes while still using two-pass estimation when available.
    psf_signature = inspect.signature(estimate_psf_from_chunks)
    if "blind_passes" in psf_signature.parameters:
        psf_estimation_kwargs["blind_passes"] = args.blind_passes
    elif args.blind_passes != 1:
        print(
            "  NOTE: this psf_estimation.py does not support blind_passes; "
            "using its single-pass chunked blind PSF estimation.",
            flush=True,
        )
    chunked_blind = estimate_psf_from_chunks(**psf_estimation_kwargs)

    print("Running pipeline-style CUDA Richardson-Lucy deconvolution...", flush=True)
    detection_na = args.detection_na if args.detection_na is not None else args.na
    pipeline_decon = deconvolve_tiff(
        image_path=crop_tiff,
        psf=chunked_blind,
        n_iters=args.lucy_iters,
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
    pipeline_decon_path = output_dir / "pipeline_cuda_DB2.tif"
    imwrite(str(pipeline_decon_path), pipeline_decon)
    print(
        f"Saved pipeline-style deconvolution output: "
        f"{pipeline_decon_path} shape={pipeline_decon.shape}",
        flush=True,
    )

    print("Estimating reference full-window blind PSF on same comparison crop/window...", flush=True)
    full_volume = open_tiff_memmap(crop_tiff)
    full_window = np.asarray(full_volume)
    print(f"Full blind window shape={full_window.shape}", flush=True)
    full_blind = _run_full_blind(
        volume=full_window,
        psf_seed=psf_seed,
        n_iters=args.blind_iters,
        pad_xy=args.pad_xy,
        pad_z=args.pad_z,
        script_dir=script_dir,
        matlab_threads=args.matlab_threads,
        matlab_timeout=args.matlab_timeout,
    )

    print("Running reference MATLAB second-pass Lucy-Richardson deconvolution...", flush=True)
    reference_decon = _run_matlab_deconvlucy(
        volume=full_window,
        psf=full_blind,
        n_iters=args.lucy_iters,
        pad_xy=args.pad_xy,
        pad_z=args.pad_z,
        script_dir=script_dir,
        matlab_threads=args.matlab_threads,
        matlab_timeout=args.matlab_timeout,
    )
    reference_decon_path = output_dir / "reference_matlab_Dec2.tif"
    imwrite(str(reference_decon_path), reference_decon)
    print(
        f"Saved reference MATLAB deconvolution output: "
        f"{reference_decon_path} shape={reference_decon.shape}",
        flush=True,
    )

    psfs = {
        "theoretical": _normalise_psf(psf_seed),
        "chunked_blind": _normalise_psf(chunked_blind),
        "full_blind": _normalise_psf(full_blind),
    }
    for name, psf in psfs.items():
        imwrite(str(output_dir / f"{name}_psf.tif"), psf.astype(np.float32, copy=False))

    pipeline_compare, reference_compare, shape_alignment = _align_decon_outputs(
        pipeline_decon,
        reference_decon,
    )
    if shape_alignment["center_cropped_for_comparison"]:
        print(
            "Deconvolution shapes differ; center-cropping both outputs to "
            f"{tuple(shape_alignment['comparison_shape_zyx'])} for metrics.",
            flush=True,
        )
        imwrite(str(output_dir / "pipeline_cuda_DB2_aligned.tif"), pipeline_compare)
        imwrite(str(output_dir / "reference_matlab_Dec2_aligned.tif"), reference_compare)

    decon_outputs = {
        "pipeline_cuda_db2": pipeline_decon,
        "reference_matlab_dec2": reference_decon,
    }
    decon_comparison_outputs = {
        "pipeline_cuda_db2": pipeline_compare,
        "reference_matlab_dec2": reference_compare,
    }
    output_paths = {
        "pipeline_cuda_db2": str(pipeline_decon_path),
        "reference_matlab_dec2": str(reference_decon_path),
        "decon_montage": str(output_dir / "decon_cross_sections.tif"),
        "psf_fwhm": str(output_dir / "psf_fwhm.tsv"),
        "psf_axis_profiles": str(output_dir / "psf_axis_profiles.tsv"),
        "psf_axis_profile_plots": str(output_dir / "psf_axis_profiles.svg"),
    }
    if shape_alignment["center_cropped_for_comparison"]:
        output_paths["pipeline_cuda_db2_aligned"] = str(output_dir / "pipeline_cuda_DB2_aligned.tif")
        output_paths["reference_matlab_dec2_aligned"] = str(output_dir / "reference_matlab_Dec2_aligned.tif")

    metrics = {
        "input_tiff": str(input_tiff),
        "comparison_crop_shape_zyx": list(cropped.shape),
        "requested_sanity_xy": args.sanity_xy,
        "resolved_comparison_xy": comparison_xy,
        "requested_blind_z_slices": args.blind_z_slices,
        "resolved_comparison_z_slices": comparison_z_slices,
        "reference_window_shape_zyx": list(full_window.shape),
        "blind_z_window": [z_window.start, z_window.stop],
        "blind_z_window_detail": z_detail,
        "outputs": output_paths,
        "shape_alignment": shape_alignment,
        "parameters": vars(args),
        "comparison_metric_parameters": {
            "correlation_coefficient": {
                "method": "Pearson correlation of flattened aligned volumes",
                "reported_fields": ["pearson_correlation", "correlation_coefficient", "ncc"],
            },
            "antspy_image_similarity": {
                "package": "antspyx",
                "import_name": "ants",
                "metrics": ["Correlation", "MeanSquares", "ANTSNeighborhoodCorrelation"],
                "reported_field": "antspy_image_similarity",
            },
            "structural_similarity": {
                "method": "Global SSIM formula over flattened aligned volumes",
                "k1": SSIM_K1,
                "k2": SSIM_K2,
                "reported_field": "ssim_global",
            },
        },
        "resolved_dxy": dxy,
        "decon_outputs": {
            name: _volume_stats(name, volume)
            for name, volume in decon_outputs.items()
        },
        "psfs": {
            name: _psf_stats(name, psf, dxy=dxy, dz=args.dz)
            for name, psf in psfs.items()
        },
        "comparisons": {
            "pipeline_cuda_db2_vs_reference_matlab_dec2": _pair_metrics(
                pipeline_compare,
                reference_compare,
            ),
        },
    }
    _write_metrics(metrics, output_dir)
    _write_psf_profile_svg(metrics, output_dir)
    imwrite(str(output_dir / "decon_cross_sections.tif"), _make_decon_montage(decon_comparison_outputs))
    print(f"Deconvolution comparison outputs written to {output_dir}", flush=True)


if __name__ == "__main__":
    main()
