function classifierData = isolateSubject(classifierData,indices)
    N = length(classifierData.activity);
    ind = 1:N;
    index = zeros(size(indices));
    
    for ii = 1:length(indices)
        index(ii) = find(ind == indices(ii));
    end
    
    ind(index) = [];
    
    classifierData.activity(ind) = [];
    classifierData.wearing(ind) = [];
    classifierData.identifier(ind) = [];
    classifierData.subject(ind) = [];
    classifierData.features(ind,:) = [];            
    classifierData.activityFrac(ind) = [];
    classifierData.subjectID(ind) = [];
    classifierData.sessionID(ind) = [];
    classifierData.states(ind) = [];
end