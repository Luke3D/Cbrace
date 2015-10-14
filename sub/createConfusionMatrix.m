function [cmat,percentCorrect,labels] = createConfusionMatrix (trueValues, predictedValues)

% creates a matrix summarizing the classification accuracy

if isnumeric(trueValues)
    % convert to cell array
    trueValues = arrayfun(@num2str,trueValues,'unif',0);
end
if isnumeric(predictedValues)
    % convert to cell array
    predictedValues = arrayfun(@num2str,predictedValues,'unif',0);
end

labels = unique({trueValues{:}, predictedValues{:}});
labels = sort(labels);
nLabels = length(labels);
cmat = zeros([nLabels nLabels]);

for i = 1:length(trueValues)
    [notUsed trueIndex] = ismember(trueValues(i), labels);
    [notUsed predictedIndex] = ismember(predictedValues(i), labels);
    cmat(trueIndex,predictedIndex) = cmat(trueIndex,predictedIndex) + 1;
end

percentCorrect = trace(cmat)/sum(sum(cmat));
% disp(['Accuracy = ' num2str(nanmean(percentCorrect) * 100) '%']);