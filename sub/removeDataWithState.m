function classifierData = removeDataWithState(classifierData,label)

ind = find(strcmp(classifierData.states,label));
classifierData.features(ind,:) = [];
classifierData.wearing(ind) = [];
classifierData.activity(ind) = [];
classifierData.identifier(ind) = [];
classifierData.subject(ind) = [];
classifierData.states(ind) = [];
classifierData.activityFrac(ind) = [];
% disp(['Removed data with location: ' label]);
end