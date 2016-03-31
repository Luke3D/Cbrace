function classifierData = getDataWithIndices(classifierData,ind)

classifierData.features(~ind,:) = [];
classifierData.wearing(~ind) = [];
classifierData.activity(~ind) = [];
classifierData.identifier(~ind) = [];
classifierData.subject(~ind) = [];
classifierData.states(~ind) = [];
classifierData.activityFrac(~ind) = [];
end