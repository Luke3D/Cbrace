function classifierData = removeDataWithNaNs(classifierData)

ind = any(isnan(classifierData.features(:,:,1)),2); %works on original data (no reflections)
classifierData.features(ind,:,:) = [];
classifierData.wearing(ind) = [];
classifierData.activity(ind) = [];
classifierData.identifier(ind) = [];
classifierData.subject(ind) = [];
classifierData.states(ind) = [];
classifierData.activityFrac(ind) = [];
classifierData.subjectID(ind) = [];
classifierData.sessionID(ind) = [];
disp(['removed ' num2str(length(find(ind))/length(ind)*100) '% of data with NaNs'])
end