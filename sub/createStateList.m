function states = createStateList(classifierData)

states = cell(length(classifierData.wearing),1);
for ind = 1:length(classifierData.wearing)
    states{ind,1} = [classifierData.wearing{ind},'/',classifierData.activity{ind}];
end
end