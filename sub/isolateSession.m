function classifierData = isolateSession(classifierData,max_sessions,min_sessions)
    
    unique_sessions = unique(classifierData.sessionID);
    N = length(unique_sessions);
    
    for ii = 1:N
        if unique_sessions(ii) > max_sessions || unique_sessions(ii) < min_sessions
            ind = find(classifierData.sessionID == unique_sessions(ii));
            
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
    end
end
