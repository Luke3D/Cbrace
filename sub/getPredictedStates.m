function [predictedStates predictedCode] = getPredictedStates(P, stateList)
[maxVals predictedCode] = max(P,[],2);
predictedStates = cell(length(predictedCode),1);
for i = 1:length(predictedCode)
    predictedStates{i} = stateList{predictedCode(i)};
end
end