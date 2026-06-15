import argparse
import subprocess
from pathlib import Path
import sys


def run_deskew(image_path, cell_name, cell_index, channels,
               timepoints, dx, dz, angle, flip, output_dir):

    script_dir = str(Path(__file__).parent.absolute())

    print(f"Running deskew with image: {image_path}, cell name: {cell_name}, "
          f"cell index: {cell_index!r}, channels: {channels}, timepoints: {timepoints}, "
          f"dx: {dx}, dz: {dz}, angle: {angle}, flip: {flip}, output_dir: {output_dir}")

    # Only set CellIndex if a non-empty value was provided.
    cell_index_line = ""
    if cell_index and str(cell_index).strip():
        cell_index_line = f"CellIndex=int32({cell_index}); "

    # Only inject ChannelsToProcess if explicitly specified (non-empty).
    # Some datasets don't use channel numbers in filenames — in that case
    # let deskew.m fall back to its own default.
    channels_line = ""
    if channels and str(channels).strip():
        channels_line = f"ChannelsToProcess=int32([{channels}]); "

    # Only inject timepoints if explicitly specified.
    timepoints_line = ""
    if timepoints and str(timepoints).strip():
        timepoints_line = f"timepoints=int32([{timepoints}]); "

    matlab_cmd = (
        f"addpath('{script_dir}'); "
        f"imagePath='{image_path}'; "
        f"CellName='{cell_name}'; "
        + cell_index_line
        + channels_line
        + timepoints_line +
        f"dx={dx}; "
        f"dz={dz}; "
        f"angle={angle}; "
        f"flip={flip}; "
        f"output_dir='{output_dir}'; "
        f"run('deskew.m');"
    )

    command = ["matlab", "-batch", matlab_cmd]

    try:
        subprocess.run(command, check=True)
    except subprocess.CalledProcessError as e:
        print(f"MATLAB execution failed with error code: {e.returncode}")
        sys.exit(1)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--image_path')
    parser.add_argument('--cell_name')
    parser.add_argument('--cell_index', default='')
    parser.add_argument('--channels', default='')
    parser.add_argument('--timepoints', default='')
    parser.add_argument('--dx')
    parser.add_argument('--dz')
    parser.add_argument('--angle')
    parser.add_argument('--flip')
    parser.add_argument('--output_dir')
    args = parser.parse_args()

    run_deskew(args.image_path, args.cell_name, args.cell_index, args.channels,
               args.timepoints, args.dx, args.dz, args.angle, args.flip, args.output_dir)
