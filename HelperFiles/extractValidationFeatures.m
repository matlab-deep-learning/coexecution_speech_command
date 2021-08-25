function values = extractValidationFeatures
% extractValidationFeatures Extract validation features from the entire
% validation dataset
%
% This function is used in both examples

% Copyright 2019-2021 The MathWorks, Inc.

[~, adsVal] = setupDatasets;

T = tall(adsVal);
featureVectorsTall = cellfun(@(x)helperExtractAuditoryFeatures(x, 16e3),T,"UniformOutput",false);
XValC = gather(featureVectorsTall);

XValidation = cat(4,XValC{:});
features = permute(XValidation,[4 3 1 2]);

YValidationC = adsVal.Labels;

values.labels = YValidationC;
values.features = features;
