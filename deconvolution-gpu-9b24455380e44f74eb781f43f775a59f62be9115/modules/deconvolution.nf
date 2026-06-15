process DECON {
    conda "${projectDir}/environment.yml"
    tag "${cell_name}"

    publishDir "${params.output_dir}/deconvolved", mode: 'copy'

    maxForks 8
    cpus 8
    memory '32 GB'
    clusterOptions '--gres=gpu:1'

    input:
    val  deskewed_dir
    val  cell_name
    val  background
    val  iter
    val  output_dir

    output:
    path "DB2_*", emit: decon_output

    script:
    // Build optional optical-parameter flags only if the user supplied them.
    // All have defaults in decon_wrapper.py so omitting is safe but not recommended
    def na_flag          = params.na          ? "--na ${params.na}"                   : ""
    def detection_na_flag = params.detection_na ? "--detection_na ${params.detection_na}" : ""
    def illumination_na_flag = params.illumination_na ? "--illumination_na ${params.illumination_na}" : ""
    def wavelength_flag  = params.wavelength  ? "--wavelength ${params.wavelength}"   : ""
    def ni_flag          = params.ni          ? "--ni ${params.ni}"                   : ""
    def ns_flag          = params.ns          ? "--ns ${params.ns}"                   : ""
    def ni0_flag         = params.ni0         ? "--ni0 ${params.ni0}"                 : ""
    def tg_flag          = params.tg          ? "--tg ${params.tg}"                   : ""
    def tg0_flag         = params.tg0         ? "--tg0 ${params.tg0}"                 : ""
    def ng_flag          = params.ng          ? "--ng ${params.ng}"                   : ""
    def ng0_flag         = params.ng0         ? "--ng0 ${params.ng0}"                 : ""
    def ti0_flag         = params.ti0         ? "--ti0 ${params.ti0}"                 : ""
    def oversample_factor_flag = params.oversample_factor ? "--oversample_factor ${params.oversample_factor}" : ""
    def psf_model_flag   = params.psf_model   ? "--psf_model ${params.psf_model}"     : ""
    def camera_pixel_size_flag = params.camera_pixel_size ? "--camera_pixel_size ${params.camera_pixel_size}" : ""
    def magnification_flag = params.magnification ? "--magnification ${params.magnification}" : ""
    def dxy_flag         = params.dxy != null ? "--dxy ${params.dxy}"                 : ""
    def dz_flag          = params.dz != null  ? "--dz ${params.dz}"                   : ""
    def psf_size_z_flag  = params.psf_size_z  ? "--psf_size_z ${params.psf_size_z}"   : ""
    def psf_size_xy_flag = params.psf_size_xy ? "--psf_size_xy ${params.psf_size_xy}" : ""
    def blind_iters_flag = params.blind_iters ? "--blind_iters ${params.blind_iters}" : ""
    def chunk_xy_flag    = params.chunk_xy    ? "--chunk_xy ${params.chunk_xy}"       : ""
    def decon_chunk_xy_flag = params.decon_chunk_xy ? "--decon_chunk_xy ${params.decon_chunk_xy}" : ""
    def pad_xy_flag      = params.pad_xy      ? "--pad_xy ${params.pad_xy}"           : ""
    def pad_z_flag       = params.pad_z != null ? "--pad_z ${params.pad_z}"           : ""
    def blind_workers_flag = params.blind_workers ? "--blind_workers ${params.blind_workers}" : ""
    def matlab_workers_flag = params.matlab_workers ? "--matlab_workers ${params.matlab_workers}" : ""
    def matlab_threads_flag = params.matlab_threads ? "--matlab_threads ${params.matlab_threads}" : ""
    def matlab_timeout_flag = params.matlab_timeout ? "--matlab_timeout ${params.matlab_timeout}" : ""
    def blind_z_slices_flag = params.blind_z_slices ? "--blind_z_slices ${params.blind_z_slices}" : ""
    def snr_weight_cap_flag = params.snr_weight_cap != null ? "--snr_weight_cap ${params.snr_weight_cap}" : ""
    def prefetch_chunks_flag = params.prefetch_chunks ? "--prefetch_chunks ${params.prefetch_chunks}" : ""
    def decon_workers_flag = params.decon_workers ? "--decon_workers ${params.decon_workers}" : ""
    def overlap_xy_flag  = params.overlap_xy  ? "--overlap_xy ${params.overlap_xy}"   : ""
    def vram_gb_flag     = params.vram_gb     ? "--vram_gb ${params.vram_gb}"         : ""
    def cache_dir_flag   = params.psf_cache_dir ? "--cache_dir ${params.psf_cache_dir}" : ""
    def no_psf_cache_flag = params.no_psf_cache ? "--no_psf_cache"                    : ""

    """
    module load cuda/11.8
    module load matlab/2024a
    export LD_LIBRARY_PATH=\${CUDA_HOME:-}/lib64:/usr/local/cuda/lib64:\${LD_LIBRARY_PATH:-}
    
    echo "=== GPU Check ==="
    nvidia-smi
    echo "================="

    python3 ${projectDir}/scripts/decon_wrapper.py \\
        --image_path "${deskewed_dir}" \\
        --channels "${params.channels}" \\
        --timepoints "${params.timepoints}" \\
        --background ${background} \\
        --iter ${iter} \\
        --script_dir "${projectDir}/scripts" \\
        ${na_flag} \\
        ${detection_na_flag} \\
        ${illumination_na_flag} \\
        ${wavelength_flag} \\
        ${ni_flag} \\
        ${ns_flag} \\
        ${ni0_flag} \\
        ${tg_flag} \\
        ${tg0_flag} \\
        ${ng_flag} \\
        ${ng0_flag} \\
        ${ti0_flag} \\
        ${oversample_factor_flag} \\
        ${psf_model_flag} \\
        ${camera_pixel_size_flag} \\
        ${magnification_flag} \\
        ${dxy_flag} \\
        ${dz_flag} \\
        ${psf_size_z_flag} \\
        ${psf_size_xy_flag} \\
        ${blind_iters_flag} \\
        ${chunk_xy_flag} \\
        ${decon_chunk_xy_flag} \\
        ${pad_xy_flag} \\
        ${pad_z_flag} \\
        ${blind_workers_flag} \\
        ${matlab_workers_flag} \\
        ${matlab_threads_flag} \\
        ${matlab_timeout_flag} \\
        ${blind_z_slices_flag} \\
        ${snr_weight_cap_flag} \\
        ${prefetch_chunks_flag} \\
        ${decon_workers_flag} \\
        ${overlap_xy_flag} \\
        ${vram_gb_flag} \\
        ${cache_dir_flag} \\
        ${no_psf_cache_flag}
    """
}
