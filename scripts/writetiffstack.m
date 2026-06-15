%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%                    
%           Function created by Bo-Jui Chang (bjo4), 2021/12/6 @ Dallas
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


function [] = writetiffstack(image, path)

    % This helper only writes 3D stacks: rows x cols x slices
    if ndims(image) ~= 3
        error('writetiffstack expects a 3D array.');
    end

    [nx, ny, nz] = size(image);
    imgType = class(image);

    % Map MATLAB class to TIFF bit depth
    switch imgType
        case {'uint8', 'int8'}
            bitsPerSample = 8;
        case {'uint16', 'int16'}
            bitsPerSample = 16;
        case {'uint32', 'int32', 'single'}
            bitsPerSample = 32;
        case {'uint64', 'int64', 'double'}
            bitsPerSample = 64;
        otherwise
            error('Unsupported image class: %s', imgType);
    end

    % Estimate output size so BigTIFF can be selected when needed
    bytesPerPixel = bitsPerSample / 8;
    estimatedSize = nx * ny * nz * bytesPerPixel;
    threshold = 4.0 * 1024^3;

    if estimatedSize > threshold
        tiffFile = Tiff(path, 'w8'); % BigTIFF
    else
        tiffFile = Tiff(path, 'w');  % Standard TIFF
    end

    tCleanup = onCleanup(@() tiffFile.close());

    % Set TIFF tags once, then write each slice as a separate directory
    tagstruct.Photometric = Tiff.Photometric.MinIsBlack;
    tagstruct.ImageLength = nx;
    tagstruct.ImageWidth = ny;
    tagstruct.PlanarConfiguration = Tiff.PlanarConfiguration.Chunky;
    tagstruct.Compression = Tiff.Compression.None;
    tagstruct.BitsPerSample = bitsPerSample;

    % Match TIFF sample format to MATLAB data type
    if contains(imgType, 'int') && ~contains(imgType, 'uint')
        tagstruct.SampleFormat = Tiff.SampleFormat.Int;
    elseif strcmp(imgType, 'single') || strcmp(imgType, 'double')
        tagstruct.SampleFormat = Tiff.SampleFormat.IEEEFP;
    else
        tagstruct.SampleFormat = Tiff.SampleFormat.UInt;
    end

    for iz = 1:nz
        tiffFile.setTag(tagstruct);
        tiffFile.write(image(:,:,iz));

        % Create a new IFD for the next slice
        if iz < nz
            tiffFile.writeDirectory();
        end
    end
end
