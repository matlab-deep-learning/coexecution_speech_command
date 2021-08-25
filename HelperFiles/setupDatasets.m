function [trainingDatastore, validationDatastore, batchSize] = setupDatasets(varargin)
% setupDatasets Create and set up the training and validation datasets

% This function is used in both examples

% Copyright 2019-2021 The MathWorks, Inc.

persistent adsTrain adsValidation miniBatchSize
if isempty(adsTrain)
    
    url = 'https://ssd.mathworks.com/supportfiles/audio/google_speech.zip';
    downloadFolder = tempdir;
    dataFolder = fullfile(downloadFolder,'google_speech');

    if ~exist(dataFolder,'dir')
        disp('Downloading data set (1.4 GB) ...')
        unzip(url,downloadFolder)
    end

    % Create datastore pointing to the entire dataset
    ads = audioDatastore(fullfile(dataFolder,'train'), ...
        'IncludeSubfolders',true, ...
        'FileExtensions','.wav', ...
        'LabelSource','foldernames');
    
    % Define the 10 commands
    commands = categorical(["yes","no","up","down","left","right","on","off","stop","go"]);
    isCommand = ismember(ads.Labels,commands);
    isUnknown = ~ismember(ads.Labels,commands);
    
    % Include 20% of remaining not-command files. Label them as "unknown"
    includeFraction = 0.2;
    mask = rand(numel(ads.Labels),1) < includeFraction;
    isUnknown = isUnknown & mask;
    ads.Labels(isUnknown) = categorical("unknown");
    
    % Create training datastore
    adsTrain = subset(ads,isCommand|isUnknown);
    adsTrain.Labels = removecats(adsTrain.Labels);
    adsTrain.Labels = double(adsTrain.Labels) - 1;
    adsTrain = shuffle(adsTrain);

    % Create validation datastore
    adsValidation = audioDatastore(fullfile(dataFolder,'validation'), ...
        'IncludeSubfolders',true, ...
        'FileExtensions','.wav', ...
        'LabelSource','foldernames');
    isUnknown = ~ismember(adsValidation.Labels,commands);
    adsValidation.Labels(isUnknown) = categorical("unknown");
    adsValidation.Labels = removecats(adsValidation.Labels);
    adsValidation.Labels = double(adsValidation.Labels) - 1;
      
    L = numel(adsTrain.Files);
    if nargin == 0
        miniBatchSize = 128;
    else
        miniBatchSize = varargin{1};
    end
    
    M = floor(L/miniBatchSize);
    adsTrain = subset(adsTrain,1:M*miniBatchSize);
end

trainingDatastore = adsTrain;
validationDatastore = adsValidation;
batchSize = miniBatchSize;
