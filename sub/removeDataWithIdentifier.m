function classifierData = removeDataWithIdentifier(classifierData,label)

ind = find(strcmp(classifierData.identifier,label));
classifierData.features(ind,:) = [];
classifierData.wearing(ind) = [];
classifierData.activity(ind) = [];
classifierData.identifier(ind) = [];
classifierData.subject(ind) = [];
classifierData.states(ind) = [];
disp(['Removed data with identifier: ' label]);
end