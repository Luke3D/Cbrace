%% CREATE CLIPS AND GET FEATURES
%New version: save date,time and clip info in feature structure
function dataPreprocessingIMU
clc, clear all, close all
addpath ./sub

parentDir = '../';
dataDir = [parentDir 'raw/'];
dirList = dir(dataDir);
patientDir = {}; controlDir = {};

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
options.overlap = 0.75;        %overlap across clips (0 to 1)
options.resample_secs = 30; %resampling rate [Hz] - Currently not used
options.datetime_columns = 2;
options.activity_columns = 2;
options.remove_ends = 0;
options.activity_fraction = 0;
options.max_rate = 40;
options.min_rate = 2;
options.forceFileRewrite = 1;

%clips directory must already exists under parent folder
clipDir = [parentDir 'clips/'];
uniqueStates = {};

%get the clips
for subject = 1:length(rawDirs)
    
    %set up data directories we are going to use
    subjectDir = rawDirs{subject};
    dataFilePrefix = [dataDir subjectDir];
    clipFile = [clipDir subjectDir '.mat'];
    
    %skip files if they already exist and we don't want to rewrite them
    if exist(clipFile,'file') && ~options.forceFileRewrite
        disp(['Skipping:' subjectDir]);
    else
        %get the clips based on our options
        [acc, date, labels, percentTimeSpent] = getTestClipsIMU(dataFilePrefix, ...
            'secs',options.secs,...
            'overlap',options.overlap,...
            'datetime_columns',options.datetime_columns,...
            'activity_columns',options.activity_columns,...
            'remove_ends',options.remove_ends,...
            'activity_fraction',options.activity_fraction,...
            'max_rate',options.max_rate,...
            'min_rate',options.min_rate);
        %begin making our clip data structure
        clip_data = [];
        clip_data.times = date;
        clip_data.acc = acc;
        clip_data.identifier = {dataFilePrefix(end)};
        clip_data.activity_fraction = percentTimeSpent;
        clip_data.samplingT = options.resample_secs;
        
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
        
        %---NOT IMPLEMENTED OR USED, IGNORE---
        for i = 1:length(uniqueStates)
            if ~any(strcmp(uniqueStates{i},uniqClipStates))
                %make a couple random clips and tack on to the end
            end
        end
        %make fake clip data by random sampling to account for classes not
        %included
        
        %need full list of possibilities
        %-------------------------------
        
        %save our results
        save(clipFile, 'clip_data');
    end
end

% calculate features for training data
files = dir(clipDir); % file names with '.' and '..'
trainingFeatureDir = [parentDir 'features/'];
for file = 1:length(files)
    filename = files(file).name;
    if files(file).isdir || filename(1) == '.' || ...
            strcmp(filename,'options.mat') || strcmp(filename,'classifierData.mat')
        continue;  % takes care of '.' and '..' and .DS_store
    end
    
    readfile = [clipDir filename];
    writefile = [trainingFeatureDir filename];
    
    if exist(writefile,'file') && ~options.forceFileRewrite
        disp(['Skipping: ' filename]);
        continue;
    else
        disp(['Calculating features: ' filename])
        load(readfile);
        
        %num_samples
        num_samples = length(clip_data.acc);
        
        %Reflections of the accelerations
        reflections = [1, 1, 1, 1];
        %             ; 1 1 -1 -1; 1 -1 1 -1; 1 -1 -1 1]; %X,Y,Z reflections
        num_reflections = size(reflections,1);
        
        act_labels = {};
        wearing_labels = {};
        identifier = {};
        subject = {};
        x_data = [];
        activity_fraction = [];
        
        for accIndex = 1:num_samples
            
            %disable reflections
            % cycle through the reflections
            %             for reflection = 1:num_reflections
%             acc_len = size(clip_data.acc{accIndex},2);
%             refl_acc = clip_data.acc{accIndex} .* repmat(reflections(reflection,:)',1,acc_len);
%             [x_vec, x_labels] = getFeatures(refl_acc,options.secs);

            [x_vec, x_labels] = getFeaturesIMU(clip_data.acc{accIndex});
            x_data = [x_data; x_vec];
            act_labels{end+1} = clip_data.act_label{accIndex};
            identifier{end+1} = clip_data.identifier{1};
            subject{end+1} = filename(1:end-4);
            activity_fraction(end+1) = clip_data.activity_fraction(accIndex);
            if ~isempty(clip_data.wearing_label)
                wearing_labels{end+1} = clip_data.wearing_label{accIndex};
            else
                wearing_labels{end+1} = 'unlabeled';
            end
            %             end
        end
        features_data.subject = subject;
        features_data.feature_labels = x_labels;
        features_data.features = x_data;
        features_data.activity_labels = act_labels;
        features_data.wearing_labels = wearing_labels;
        features_data.identifier = identifier;
        features_data.activity_fraction = activity_fraction;
        features_data.times = clip_data.times';
        features_data.samplingT = options.resample_secs;
        features_data.clipdur = options.secs;
        features_data.clipoverlap = options.overlap;
        features_data.recordtime = (round( (1-options.overlap)*size(x_data,1) ) + 1)*options.secs;
        
        save(writefile, 'features_data');
    end
end