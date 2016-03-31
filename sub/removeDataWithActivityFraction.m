function [classifierData,ind] = removeDataWithActivityFraction(classifierData,threshold)

ind = find(classifierData.activityFrac < threshold);
classifierData.features(ind,:) = [];
classifierData.wearing(ind) = [];
classifierData.activity(ind) = [];
classifierData.identifier(ind) = [];
classifierData.subject(ind) = [];
classifierData.states(ind) = [];
classifierData.activityFrac(ind) = [];
% disp(['Removed data with location: ' label]);
end