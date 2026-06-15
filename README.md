# deconvolution-pipeline

A Nextflow DSL2 pipeline for GPU-accelerated deskewing and deconvolution of light-sheet microscopy (ctASLM) TIFF volumes. The pipeline runs on SLURM and uses a conda environment managed by **mamba**. It requires **Java 17+** and **mamba** to be available on the cluster. Made for BioHPC @ UTSouthwestern. 

---

## Requirements

| Requirement | Version / Notes |
|---|---|
| Java | 17 or later (required by Nextflow) |
| Nextflow | Must be loadable from the cluster module system |
| Mamba | Must be loadable via `module load mamba` |
| SLURM | Pipeline submits jobs to `super` (DESKEW) and `GPU` (DECON) queues |
| CUDA | 11.8 (loaded via `module load cuda/11.8`) |
| MATLAB | 2024a (loaded via `module load matlab/2024a`) |
| GPU | 1× NVIDIA GPU per DECON job (`--gres=gpu:1`) |

The conda environment (`decon_env`) is built automatically from `environment.yml` the first time the pipeline runs. Key packages: `pycudadecon 0.5.1`, `cudatoolkit 11.8`, `dask`, `tifffile`, `numpy`, `psfmodels`, `antspyx`.

---

## Pipeline Process

The pipeline has two sequential stages:

### 1. DESKEW

Calls MATLAB 2024a (`deskew.m`) via a Python wrapper (`deskew_wrapper.py`) to correct the oblique acquisition angle of ctASLM data. It applies a 3-D shear transform (`imrotate3`) to each selected channel/timepoint TIFF and writes output into two folders:

- `<output_dir>/shear/` — intermediate sheared volumes
- `<output_dir>/Top_shear/` — final deskewed volumes (passed to DECON)

### 2. DECON

Reads the deskewed `CH*.tif` files from `Top_shear/` and runs Richardson–Lucy GPU deconvolution via `pycudadecon`. Volumes are processed as full-Z XY tiles using Dask `map_overlap` with reflect-padded boundaries to suppress edge artifacts. Output files are named `DB2_<original_stem>.tif` and written to `<output_dir>/deconvolved/`.

**PSF resolution always uses blind estimation.** The pipeline generates a theoretical Gibson-Lanni PSF from the optical parameters you supply, but uses it only as the starting guess for MATLAB `deconvblind`. It then estimates the PSF from the first TIFF by tiling it into XY chunks, running `deconvblind` on each tile, and merging the per-tile PSFs with SNR weighting. The merged blind PSF is saved as `estimated_psf.tif` alongside the input TIFFs and cached for reuse.

---

## Running the Pipeline

```bash
nextflow run main.nf -profile my_cluster \
    --image_path /path/to/raw/tiffs \
    --cell_name MyCellName \
    --output_dir /path/to/output
```

Add `-resume` to restart from the last successful checkpoint after a failure.

To skip deskewing and run deconvolution on already-deskewed data:

```bash
nextflow run main.nf -profile my_cluster \
    --decon_only true \
    --decon_input_dir /path/to/Top_shear \
    --cell_name MyCellName \
    --output_dir /path/to/output
```

To compare the pipeline deconvolution against the reference MATLAB
`deconvblind -> deconvlucy` flow on already-deskewed data:

```bash
nextflow run main.nf -profile compare_psf \
    --decon_input_dir /path/to/Top_shear \
    --output_dir /path/to/output
```

Comparison outputs are published to `<output_dir>/comparison/`.

---

## Parameters

All parameters can be passed on the command line as `--param_name value` or set in a custom `nextflow.config`.

### Required

| Parameter | Description |
|---|---|
| `--image_path` | Path to the directory containing raw input TIFFs |
| `--cell_name` | Name prefix used to locate and label files |

### I/O

| Parameter | Default | Description |
|---|---|---|
| `--output_dir` | `output` | Root directory for all outputs |
| `--decon_only` | `false` | Skip deskewing; go straight to deconvolution |
| `--decon_input_dir` | `''` | Input directory for `--decon_only` mode (overrides `--image_path`) |
| `--tiff_index` | `0` | Matching input TIFF index used by the comparison workflow |

### Deskew

| Parameter | Default | Description |
|---|---|---|
| `--cell_index` | `''` | Integer index to select a specific cell in the dataset |
| `--channels` | `''` | Channels to process, e.g. `0` or `0,1,2`. Empty = all channels |
| `--timepoints` | `''` | Timepoints to process, e.g. `0` or `0,1`. Empty = all timepoints |
| `--dx` | `0.118` | Lateral pixel size in µm |
| `--dz` | `0.118` | Axial step size in µm |
| `--angle` | `40` | Acquisition angle in degrees |
| `--flip` | `1` | Flip direction flag (1 or -1) |

### Deconvolution

| Parameter | Default | Description |
|---|---|---|
| `--iter` | `10` | Number of Richardson–Lucy iterations |
| `--background` | `0` | Background value subtracted before deconvolution |

### Blind PSF Estimation

| Parameter | Default | Description |
|---|---|---|
| `--blind_iters` | `10` | MATLAB `deconvblind` iterations per chunk |
| `--chunk_xy` | `256` | XY tile size for blind estimation (px). `<=0` auto-sizes from VRAM |
| `--pad_xy` | `32` | XY halo added per edge before each blind chunk (px) |
| `--pad_z` | `20` | Z halo added per edge before each blind chunk (slices) |
| `--blind_z_slices` | `128` | Z planes used per blind PSF tile. `<=0` uses full Z |
| `--blind_workers` | `8` | Concurrent blind PSF chunk workers |
| `--matlab_workers` | `8` | Concurrent MATLAB `deconvblind` processes (keep `1` on SLURM) |
| `--matlab_threads` | `1` | Threads per MATLAB process (clamped to 1–2) |
| `--matlab_timeout` | `1800` | Seconds before a blind chunk is killed. `<=0` disables |
| `--snr_weight_cap` | `100` | Max per-chunk SNR weight during PSF merge; prevents bright-artifact dominance |
| `--prefetch_chunks` | `0` | PSF tile read-ahead. `<=0` = one worker batch |
| `--psf_cache_dir` | `''` | Directory to cache/reuse blind PSF estimates |
| `--no_psf_cache` | `false` | Disable PSF cache; always re-estimate |

### CUDA Deconvolution Chunking

| Parameter | Default | Description |
|---|---|---|
| `--decon_chunk_xy` | `0` | Core XY tile size for CUDA decon (px). `<=0` auto-sizes from VRAM |
| `--overlap_xy` | `0` | XY overlap between tiles (px). `<=0` = PSF-size/4, capped at 48 |
| `--decon_workers` | `1` | Dask workers for CUDA decon chunks |
| `--vram_gb` | `0` | Override detected free VRAM (GiB) for auto-sizing |

### Optical / PSF Parameters

These are used to generate the theoretical PSF seed for blind estimation. The theoretical PSF is not used directly for final deconvolution. All are optional with defaults.

| Parameter | Default | Description |
|---|---|---|
| `--na` | `1.0` | Detection numerical aperture (backward-compatible) |
| `--detection_na` | `''` | Detection NA; overrides `--na` when provided |
| `--illumination_na` | `''` | Illumination NA (metadata only; not used by `psfmodels`) |
| `--wavelength` | `0.520` | Emission wavelength in µm |
| `--ni` | `1.515` | Immersion medium refractive index |
| `--ns` | `''` | Sample refractive index |
| `--ni0` | `''` | Design immersion refractive index |
| `--tg` | `''` | Experimental coverslip thickness (µm) |
| `--tg0` | `''` | Design coverslip thickness (µm) |
| `--ng` | `''` | Experimental coverslip refractive index |
| `--ng0` | `''` | Design coverslip refractive index |
| `--ti0` | `''` | Objective working distance (µm) |
| `--dxy` | `0.118` | Lateral pixel size used for PSF model (µm) |
| `--psf_size_z` | `101` | Z dimension of PSF volume (voxels) |
| `--psf_size_xy` | `61` | XY dimension of PSF volume (voxels) |
| `--psf_model` | `vectorial` | PSF model type: `vectorial`, `scalar`, or `gaussian` |
| `--oversample_factor` | `3` | PSF model oversampling factor |
| `--camera_pixel_size` | `''` | Camera pixel size (µm); used to derive `dxy` when `--dxy <= 0` |
| `--magnification` | `''` | Total magnification; used to derive `dxy` when `--dxy <= 0` |

---

## Output Structure

```
<output_dir>/
├── shear/              # Intermediate sheared volumes (from DESKEW)
├── Top_shear/          # Deskewed volumes passed to DECON
│   ├── CH0_0.tif
│   ├── estimated_psf.tif   # Merged blind PSF used for deconvolution
│   └── ...
└── deconvolved/        # Final deconvolved TIFFs
    ├── DB2_CH0_0.tif
    └── ...
```

Comparison runs publish to `<output_dir>/comparison/`:

```
comparison/
├── pipeline_cuda_DB2.tif
├── reference_matlab_Dec2.tif
├── decon_metrics.json
├── decon_metrics.tsv         # Pearson, SSIM-style, and ANTsPy similarity metrics
├── decon_cross_sections.tif
├── psf_fwhm.tsv             # X/Y/Z Gaussian-fit and half-max FWHM per PSF
├── psf_axis_profiles.tsv    # Raw and fitted X/Y/Z PSF line profiles
├── psf_axis_profiles.svg    # Plotted raw and fitted PSF line profiles
├── chunked_blind_psf.tif
├── full_blind_psf.tif
└── theoretical_psf.tif
```

---

## Notes

- Load the cluster Nextflow module before running the pipeline.
- The pipeline profile `my_cluster` enables conda/mamba. The `docker` profile is also available for non-HPC use.
- DESKEW runs on the `super` queue, while DECON runs on the `GPU` queue.
- The active deconvolution PSF is always the merged blind estimate. The theoretical PSF generated from optical parameters is only a starting guess for blind estimation.
- Nextflow work directories accumulate large intermediate files. Clean up with `nextflow clean -f` after a successful run.
