process DESKEW {
    
    // Directives go first (before input/output/script)
    publishDir { output_dir }, mode: 'copy'

    input:
    val image_path
    val cell_name
    val cell_index
    val channels
    val timepoints
    val dx
    val dz
    val angle
    val flip
    val output_dir

    output:
    val "${output_dir}/Top_shear", emit: deskewed_path

    script:
    """
    module load matlab/2024a

    python3 ${projectDir}/scripts/deskew_wrapper.py \
        --image_path "${image_path}" \
        --cell_name "${cell_name}" \
        --cell_index "${cell_index}" \
        --channels "${channels}" \
        --timepoints "${timepoints}" \
        --dx ${dx} \
        --dz ${dz} \
        --angle ${angle} \
        --flip ${flip} \
        --output_dir "${output_dir}"
    """
}
