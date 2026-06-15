%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%                    
%           Function created by Bo-Jui Chang (bjo4), 2021/12/6 @ Dallas
%           Modified to support BigTIFF and dynamic data types
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


function [FinalImage, mImage, nImage, NumberImages] = readtiffstack(path)

    % Validate input path before trying to read metadata
    if ~isfile(path)
        error('TIFF file does not exist: %s', path);
    end

    InfoImage = imfinfo(path);
    NumberImages = numel(InfoImage);

    if NumberImages == 0
        error('No TIFF pages found: %s', path);
    end

    % Image size is taken from the first page
    mImage = InfoImage(1).Height;
    nImage = InfoImage(1).Width;

    % Determine the MATLAB class from TIFF metadata
    bitsPerSample = InfoImage(1).BitsPerSample;

    if isfield(InfoImage(1), 'SampleFormat')
        sampleFormat = InfoImage(1).SampleFormat; % 1=uint, 2=int, 3=float
    else
        sampleFormat = 1; % Default to unsigned integer
    end

    if sampleFormat == 3
        if bitsPerSample == 32
            dataType = 'single';
        else
            dataType = 'double';
        end
    elseif sampleFormat == 2
        if bitsPerSample == 8
            dataType = 'int8';
        elseif bitsPerSample == 16
            dataType = 'int16';
        elseif bitsPerSample == 32
            dataType = 'int32';
        else
            dataType = 'int64';
        end
    else
        if bitsPerSample == 8
            dataType = 'uint8';
        elseif bitsPerSample == 16
            dataType = 'uint16';
        elseif bitsPerSample == 32
            dataType = 'uint32';
        else
            dataType = 'uint64';
        end
    end

    % Preallocate output stack using the TIFFs native numeric type
    FinalImage = zeros(mImage, nImage, NumberImages, dataType);

    % Suppress noisy TIFF warnings, but restore warning state even on error
    warnState = warning('off', 'all');
    c = onCleanup(@() warning(warnState));

    TifLink = Tiff(path, 'r');
    tCleanup = onCleanup(@() TifLink.close());

    for i = 1:NumberImages
        TifLink.setDirectory(i);
        page = TifLink.read();

        % Fail fast if one page does not match the expected stack size
        if ~isequal(size(page,1), mImage) || ~isequal(size(page,2), nImage)
            error('Page %d size mismatch in %s', i, path);
        end

        FinalImage(:,:,i) = page;
    end
end