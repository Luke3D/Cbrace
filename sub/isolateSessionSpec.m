function classifierData = isolateSessionSpec(classifierData,IDs,brace)
    
    %Find indices for given brace in classifierData
    idx = strmatch(brace,classifierData.subjectBrace,'exact');
    ID_brace = classifierData.sessionID(idx); %session IDs of brace of interest

    unique_sessions = unique(ID_brace);
    N = length(unique_sessions);
    ID_list = 1:N;
    ID_list(sort(IDs)) = []; %array of session IDs to remove
    
    %Cycle through and remove sessions the user does not want
    ind = [];
    for ii = 1:length(ID_list)
        idx2 = find(ID_brace == ID_list(ii));
        ind = [ind; idx(idx2)]; %convert to indexing of original classifierData
    end

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
