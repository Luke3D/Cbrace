function classifierData = isolateBrace(classifierData,brace)
    N = length(classifierData.activity);
    ind = 1:N;

    if strcmpi(brace,'both') %analyze both braces
        index1 = strmatch('Cbr',classifierData.subjectBrace,'exact');
        index2 = strmatch('SCO',classifierData.subjectBrace,'exact');
        
        ind([index1; index2]) = [];
        
        classifierData.activity(ind) = [];
        classifierData.wearing(ind) = [];
        classifierData.identifier(ind) = [];
        classifierData.subject(ind) = [];
        classifierData.features(ind,:) = [];            
        classifierData.activityFrac(ind) = [];
        classifierData.subjectID(ind) = [];
        classifierData.sessionID(ind) = [];
        classifierData.states(ind) = []; 
        classifierData.subjectBrace(ind) = []; 
    else %analyze CBR or SCO
        index = strmatch(brace,classifierData.subjectBrace,'exact');        
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
        classifierData.subjectBrace(ind) = []; 
    end
end