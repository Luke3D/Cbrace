function classifierData = combineLocations(classifierData,label)

for i = 1:length(classifierData.wearing)
   if strfind(classifierData.wearing{i},label)
       classifierData.wearing{i} = label;
   end
end
% disp(['Combined locations: ' label]);
end