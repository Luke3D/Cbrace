function classifierData = replaceStateWithState(classifierData,old,new)

ind = find(strcmp(classifierData.states,old));
% classifierData.features(ind,:) = [];

% classifierData.identifier(ind) = [];
% classifierData.subject(ind) = [];

stateStr = regexp(new,'/','split');
for i  = 1:length(ind)
    classifierData.states{ind(i)} = new;
    classifierData.wearing{ind(i)} = stateStr{1};
    classifierData.activity{ind(i)} = stateStr{2};
end

% classifierData.activityFrac(ind) = [];
% disp(['Removed data with location: ' label]);
end