function classifierData = removeDataWithSubject(classifierData,label)

ind = find(strcmp(classifierData.subject,label));
classifierData.features(ind,:) = [];
classifierData.wearing(ind) = [];
classifierData.activity(ind) = [];
classifierData.identifier(ind) = [];
classifierData.subject(ind) = [];
classifierData.states(ind) = [];
end