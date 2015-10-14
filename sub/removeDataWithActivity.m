function classifierData = removeDataWithActivity(classifierData,label)

ind = find(strcmp(classifierData.activity,label));
classifierData.features(ind,:,:) = [];
classifierData.wearing(ind) = [];
classifierData.activity(ind) = [];
classifierData.identifier(ind) = [];
classifierData.subject(ind) = [];
classifierData.states(ind) = [];
classifierData.activityFrac(ind) = [];
classifierData.subjectID(ind) = [];
classifierData.sessionID(ind) = [];
%12/5/2013 Fixes bug
% disp(['Removed data with activity: ' label]);
end