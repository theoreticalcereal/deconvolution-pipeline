%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%           Deskew the data by simply shifting the images laterally 
%           Currently working on the images that move up and down (not left
%           and right). If you want to process the data left and right,
%           please rotate the data first.
%           Note: 
%           1. You can choose if you want to save the deskew(shear) images.
%           2. BigTIFF format is automatically used for files larger than 4GB,
%           preserving full resolution for all output images.
%           3. Fixed some bugs about artificial strips in the topview. It
%           was discovered by Tadamoto Isogai
%           Bo-Jui, 2025/5/9 @ Dallas
%           Updated 2025/12/3: Added BigTIFF support to preserve full resolution
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


%% Configuration Parameters


% =========================================================================
% INJECTED BY NEXTFLOW / PYTHON WRAPPER
% The following variables are passed directly into the workspace:
% imagePath, CellName, CellIndex, dx, dz, angle, flip, output_dir
% DO NOT hardcode them here or it will break the pipeline
% =========================================================================

if ~exist('timepoints', 'var')
    timepoints = [];
end

if ~exist('output_dir', 'var') || isempty(output_dir)
    error('output_dir was not provided by the wrapper.');
end


tic;

%% Processing Setup
numFolders = 1;

for c = 1:numFolders

    % Use the folder name passed from Nextflow directly.
    cellNameWithIndex = CellName;

    inputDir = fullfile(imagePath, cellNameWithIndex);

    % Discover input files in the cell folder.
    tifFiles = dir(fullfile(inputDir, '*.tif'));
    if isempty(tifFiles)
        tifFiles = dir(fullfile(inputDir, '*.tiff'));
    end
    if isempty(tifFiles)
        error('No TIFF files found in %s', inputDir);
    end

    % Sort files for deterministic processing.
    [~, idx] = sort({tifFiles.name});
    tifFiles = tifFiles(idx);

    detectedChannels = [];
    detectedTimepoints = [];
    for i = 1:numel(tifFiles)
        tokens = regexp(tifFiles(i).name, '^CH(\d+)_(\d+)(?:_registered_consistent)?\.tiff?$', 'tokens', 'once');
        if ~isempty(tokens)
            detectedChannels(end + 1) = str2double(tokens{1}); %#ok<SAGROW>
            detectedTimepoints(end + 1) = str2double(tokens{2}); %#ok<SAGROW>
        end
    end

    if isempty(detectedChannels)
        error('No CH##_###### TIFF files found in %s', inputDir);
    end

    if ~exist('ChannelsToProcess', 'var') || isempty(ChannelsToProcess)
        ChannelsToProcess = unique(detectedChannels);
    end
    if isempty(timepoints)
        timepoints = unique(detectedTimepoints);
    end

    disp(sprintf('Detected channels: %s', mat2str(unique(detectedChannels))));
    disp(sprintf('Detected timepoints: %s', mat2str(unique(detectedTimepoints))));
    disp(sprintf('Processing channels: %s', mat2str(ChannelsToProcess)));
    disp(sprintf('Processing timepoints: %s', mat2str(timepoints)));

    % Process each detected/requested timepoint and channel
    for ti = 1:numel(timepoints)
        t = timepoints(ti);
        for ch = 1:numel(ChannelsToProcess)
            tic;

            % Build the expected filename for this channel/timepoint.
            baseName = sprintf('CH%02d_%06d', ChannelsToProcess(ch), t);
            registeredBaseName = sprintf('%s_registered_consistent', baseName);

            % Try both supported extensions.
            candidates = {
                fullfile(inputDir, [baseName '.tif'])
                fullfile(inputDir, [baseName '.tiff'])
                fullfile(inputDir, [registeredBaseName '.tif'])
                fullfile(inputDir, [registeredBaseName '.tiff'])
            };

            filepath = '';
            for k = 1:numel(candidates)
                if isfile(candidates{k})
                    filepath = candidates{k};
                    break;
                end
            end

            if isempty(filepath)
                error('Missing expected file: %s', baseName);
            end

            Image = readtiffstack(filepath);

            % Image dimensions
            [ysize, xsize, zsize] = size(Image);
            newdz = dz * cosd(angle);

            disp("Deskewing ... ");
            clear ShearImage
            cz = floor(zsize / 2) + 1;

            max_yoffset = abs(round(flip * (1 - cz) * (newdz / dx)));
            ShearImage = zeros(ysize + 2 * max_yoffset, xsize, zsize, 'uint16');


            tic
            for z = 1:zsize
                yoffset = round(flip * (z - cz) * (newdz / dx));
                ShearImage(yoffset + max_yoffset + 1 : ysize + yoffset + max_yoffset, :, z) = Image(:,:,z);
            end
            toc

            % Average adjacent slices to reduce artifact ringing in the top view.
            for z = 1:zsize-1
                ShearImage(:,:,z) = (ShearImage(:,:,z) + ShearImage(:,:,z+1)) / 2;
            end

            toc

            % Save the processed image stack
            disp("Saving the shear image");
            tic
            output_size = size(ShearImage,1) * size(ShearImage,2) * size(ShearImage,3) * 2 / (1024*1024*1024);

            outputFolder = fullfile(output_dir, 'shear');
            if ~isfolder(outputFolder)
                mkdir(outputFolder);
            end
            if output_size > 4
                disp(sprintf("File is larger than 4GB (%.2f GB), saving as BigTIFF format", output_size));
            end
            writetiffstack(ShearImage, fullfile(outputFolder, [baseName '.tif']));
            toc

            %% Rotation to top view
            disp("Rotating to top view ... ");
            clear mipzy scaled_ShearImage scaled_mipzy rot_scaled_mipzy mask cropped_mipzy zy_view rotTop_ShearImage;
            
            % Compute MIP on the 3rd dimension (correct for 2D display)
            mipzy = max(ShearImage, [], 3);

            figure(1)
            imagesc(mipzy); axis equal tight

            scale_x = dz * sind(angle) / dx;

            % Resize only in x-direction using single precision to save memory
            scaled_ShearImage = imresize3(single(ShearImage), ...
                [size(ShearImage,1), size(ShearImage,2), round(size(ShearImage,3) * scale_x)], ...
                'Method', 'linear');

            scaled_mipzy = max(scaled_ShearImage, [], 3);

            figure(2)
            imagesc(scaled_mipzy); axis equal tight

            rot_scaled_mipzy = imrotate(scaled_mipzy, -1 * flip * angle, 'bilinear', 'crop');
            figure(2)
            imagesc(rot_scaled_mipzy); axis equal tight

            % Rotate entire 3D volume at once using imrotate3 (much faster than slice-by-slice)
            tic
            rotTop_ShearImage = imrotate3(scaled_ShearImage, -1 * flip * angle, [0 0 1], 'nearest', 'crop');
            rotTop_ShearImage = uint16(rotTop_ShearImage);
            toc

            rotTop_ShearImage = permute(rotTop_ShearImage, [1 3 2]);

            output_size_rotTop = size(rotTop_ShearImage,1) * size(rotTop_ShearImage,2) * size(rotTop_ShearImage,3) * 2 / (1024*1024*1024);

            outputFolder2 = fullfile(output_dir, 'Top_shear');
            if ~isfolder(outputFolder2)
                mkdir(outputFolder2);
            end

            disp("Saving the top-view image");
            if output_size_rotTop > 4
                disp("File is larger than 4GB, saving as BigTIFF format");
            end

            tic
            writetiffstack(rotTop_ShearImage, fullfile(outputFolder2, [baseName '.tif']));

            % Write note file
            msg = 'z pixel = x(y) pixel. Full resolution preserved.';
            if output_size_rotTop > 4
                msg = sprintf('%s BigTIFF format used for %.2f GB file.', msg, output_size_rotTop);
            end

            note_filename = fullfile(outputFolder2, 'note.txt');
            fileID = fopen(note_filename,'w');
            fprintf(fileID,'%s', msg);
            fclose(fileID);
            toc;
            toc;
        end
    end
end

toc
disp("All processing completed successfully.");
