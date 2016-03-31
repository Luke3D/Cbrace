function dataFilenames = acquireDataDir(dataDir)
dirList = dir(dataDir);

patientDir = {};
controlDir = {};

for directory = 1:length(dirList)
    dirName = dirList(directory).name;
    
    %skip hidden files
    if dirName(1) == '.'
        continue
    elseif strcmp(dirName(end),'p')
        patientDir{end+1} = dirName;
    elseif strcmp(dirName(end),'c')
        controlDir{end+1} = dirName;
    end
end
dataFilenames = {patientDir{:}, controlDir{:}};
end