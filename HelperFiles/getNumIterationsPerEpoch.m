function numIterations = getNumIterationsPerEpoch
% getNumIterationsPerEpoch Get the number of mini-batches per epoch
%
% This function is called from the Python code in the 'Call ML from Python'
% example

% Copyright 2019-2020 The MathWorks, Inc.

[trainingDatastore, ~, batchSize] = setupDatasets;
numIterations = numel(trainingDatastore.Files)/batchSize;

end
