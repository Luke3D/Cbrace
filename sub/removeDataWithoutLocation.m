function classifierData = removeDataWithoutLocation(classifierData,label)

ind = find(~strcmp(classifierData.wearing,label));
classifierData.features(ind,:,:) = [];
classifierData.wearing(ind) = [];
classifierData.activity(ind) = [];
classifierData.identifier(ind) = [];
classifierData.subject(ind) = [];
classifierData.states(ind) = [];
classifierData.activityFrac(ind) = [];
end