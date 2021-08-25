function values = extractTrainingFeatures
% extractTrainingFeatures Extract one minibatch of training features
%
% This function is used in the 'Call ML from Python' example. The function
% is called from inside Python.

% Copyright 2019-2020 The MathWorks, Inc.

persistent adsTrain augmenter fileIdx  miniBatchSize numFiles

if isempty(adsTrain)
    
    [adsTrain, ~, miniBatchSize] = setupDatasets;
    numFiles = numel(adsTrain.Files);
   
    % Data augmenter (pitch shifter)
    augmenter = audioDataAugmenter('TimeStretchProbability', 0,...
                                   'VolumeControlProbability', 0,...
                                   'AddNoiseProbability', 0,...
                                   'TimeShiftProbability', 0,...
                                   'PitchShiftProbability',0.75,...
                                   'SemitoneShiftRange',[-8 8]);

    fileIdx = 1;
end

if fileIdx > numFiles
    adsTrain = shuffle(adsTrain);
    fileIdx = 1;
end

% Extract a minibatch of features in parallel using tall (PCT)
adsSub = subset(adsTrain,fileIdx:fileIdx+miniBatchSize-1);
T = tall(adsSub);
augmentedTall = cellfun(@(x)augmentData(x,augmenter),T,"UniformOutput",false);
featureVectorsTall = cellfun(@(x)helperExtractAuditoryFeatures(x, 16e3),augmentedTall,"UniformOutput",false);
[~,XTrainC] = evalc('gather(featureVectorsTall)');
YTrainC = adsTrain.Labels(fileIdx:fileIdx+miniBatchSize-1);

fileIdx = fileIdx + miniBatchSize;

XTrain = cat(4,XTrainC{:});
features = permute(XTrain,[4 3 1 2]);

labels = YTrainC;

values.labels = labels;
values.features = features;

% -------------------------------------------------------------------------
function y = augmentData(x,augmenter)
x = single(x);
results  = augment(augmenter,x, 16e3);
y = results.Audio{1};