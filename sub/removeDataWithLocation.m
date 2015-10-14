function classifierData = removeDataWithLocation(classifierData,label)

ind = find(strcmp(classifierData.wearing,label));
classifierData.features(ind,:,:) = [];
classifierData.wearing(ind) = [];
classifierData.activity(ind) = [];
classifierData.identifier(ind) = [];
classifierData.subject(ind) = [];
try
    classifierData.states(ind) = [];
catch
end
classifierData.activityFrac(ind) = [];
classifierData.subjectID(ind) = [];
classifierData.sessionID(ind) = [];
% disp(['Removed data with location: ' label]);
end