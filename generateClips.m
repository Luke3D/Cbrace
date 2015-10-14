%function generateClips
%Gather all file names in folder "raw"
addpath ./sub
parentDir = '../';

%Create "clips" folder string
proceed = 1;
while proceed > 0
    population = input('Are you analyzing healthy or patient or home? ','s');
    if strcmpi(population,'patient')
        clipDir = [parentDir 'clips_patient/'];
        dataDir = [parentDir 'raw/'];
        proceed = 0;
    elseif strcmpi(population,'healthy')
        clipDir = [parentDir 'clips_healthy/'];
        dataDir = [parentDir 'raw/'];
        proceed = 0;
    elseif strcmpi(population,'home')
        clipDir = [parentDir 'clips_home/'];
        dataDir = [parentDir 'home_labeled/'];
        proceed = 0;
    else
        disp('Please type healthy or patient or home.');
        proceed = 1;
    end
end
dirList = dir(dataDir);
patientDir = {}; controlDir = {};

%For each file look for "p" and "c" at end of file names
for directory = 1:length(dirList)
    dirName = dirList(directory).name;
    %skip hidden files
    if dirName(1) == '.'
        continue
    elseif strcmp(dirName(end),'p') %patient data
        patientDir{end+1} = dirName;
    elseif strcmp(dirName(end),'c') %control data
        controlDir{end+1} = dirName;
    end
end
rawDirs = {patientDir{:}, controlDir{:}};

filePathName = 'filepath';
%Set options for training clips
options.secs = 6;           %clip duration [s]
options.overlap = 0.75;        %percentage overlap across clips (0 to 1)
options.resample_secs = 30; %resampling rate [Hz] - Currently not used 
options.datetime_columns = 2;
options.activity_columns = 2;
options.remove_ends = 0;
options.activity_fraction = 0;
options.max_rate = 40;
options.min_rate = 2;
options.forceFileRewrite = 1;


uniqueStates = {};

%Write to "clips" folder + go through every folder marked with "p" and "c"
for subject = 1:length(rawDirs)
    
    %Set up data directory for mat file
    subjectDir = rawDirs{subject};
    dataFilePrefix = [dataDir subjectDir];
    clipFile = [clipDir subjectDir '.mat'];
    
    %Skip files if they already exist and we don't want to rewrite them
    if exist(clipFile,'file') && ~options.forceFileRewrite
        disp(['Skipping:' subjectDir]);
    else %File does not exist and we proceed
        %get the clips based on our options
        [acc, date, labels, percentTimeSpent, num_clips] = getTestClips(dataFilePrefix, ...
            'secs',options.secs,...
            'overlap',options.overlap,...
            'datetime_columns',options.datetime_columns,...
            'activity_columns',options.activity_columns,...
            'remove_ends',options.remove_ends,...
            'activity_fraction',options.activity_fraction,...
            'max_rate',options.max_rate,...
            'min_rate',options.min_rate);
        
        %Clip data structure
        clip_data = [];
        clip_data.times = date;
        clip_data.acc = acc;
        clip_data.identifier = {dataFilePrefix(end)};
        clip_data.activity_fraction = percentTimeSpent;
        clip_data.samplingT = options.resample_secs;
        
        %Session ID for each clip
        sessions = length(num_clips);
        sessionID = [];
        for ii = 1:sessions
            sessionID = [sessionID repmat(ii,1,num_clips(ii))];
        end
        clip_data.sessionID = sessionID;
        
        %current options for labels are:
        %{1} activity labels
        %{2} how the user is wearing label
        %if the wearing label is empty
        if size(labels,1) == 1
            clip_data.act_label = labels{1};
            clip_data.wearing_label = {};
        elseif size(labels,1) == 2
            clip_data.act_label = labels{1};
            clip_data.wearing_label = labels{2};
        end
        for i = 1:length(clip_data.act_label)
            clip_data.states{i} = [clip_data.wearing_label{i} '/' clip_data.act_label{i}];
        end
        uniqClipStates = unique(clip_data.states);
        
        %Create mat file with all the data for each "p" or "c" file
        save(clipFile,'clip_data','options');
    end
end
%end