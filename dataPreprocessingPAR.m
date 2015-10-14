%% CREATE CLIPS AND GET FEATURES
%function dataPreprocessingPAR
%SPECIFY WHICH FILES TO CREATE FEATURES FOR USING _p IN RAW FOLDER

clc, clear all, close all
addpath ./sub
parentDir = '../';

%Directory Infor + Create "clips" folder string
proceed = 1;
while proceed > 0
    population = input('Are you analyzing healthy or patient or home? ','s');
    if strcmpi(population,'patient')
        dataDir = [parentDir 'raw/'];
        clipDir = [parentDir 'clips_patient/'];
        trainingFeatureDir = [parentDir 'features_patient/']; %"features" folder string
        proceed = 0;
    elseif strcmpi(population,'healthy')
        dataDir = [parentDir 'raw/'];
        clipDir = [parentDir 'clips_healthy/'];
        trainingFeatureDir = [parentDir 'features_healthy/']; %"features" folder string
        proceed = 0;
    elseif strcmpi(population,'home')
        dataDir = [parentDir 'home_labeled/'];
        clipDir = [parentDir 'clips_home/'];
        trainingFeatureDir = [parentDir 'features_home/']; %"features" folder string
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

%Calculate features for training data
files = dir(clipDir); % file names with '.' and '..'

for file = 1:length(rawDirs)
    fileID = rawDirs(file);
    filename = char(strcat(fileID,'.mat'));
    
    readfile = char(strcat(clipDir,filename));
    writefile = char(strcat(trainingFeatureDir,filename));
    
    if exist(writefile,'file')
        disp(['Skipping: ' filename]);
        disp('Features .mat file already exists.');
        continue;
    else
        disp(['Calculating features for ' char(fileID)])
        disp(filename)
        clip_data = struct();
        load(readfile);
        
        % num_samples counting all the reflections
        num_samples = length(clip_data.times);
        
        act_labels = cell(num_samples,1);
        wearing_labels = cell(num_samples,1);
        identifier = cell(num_samples,1);
        subject = cell(num_samples,1);
        activity_fraction = zeros(num_samples,1);
        X_total = [];
        
        %Reflections of the accelerations
        reflections = [1, 1, 1, 1];  
        num_reflections = size(reflections,1);
        
        %Open up a parallel pool on the local machine
        p = gcp('nocreate');
        if isempty(p)
            parpool('local')
        end
        
        for accIndex = 1:num_samples
            act_labels{accIndex} = clip_data.act_label{accIndex};
            identifier{accIndex} = clip_data.identifier{1};
            subject{accIndex} = filename(1:end-4);
            activity_fraction(accIndex) = clip_data.activity_fraction(accIndex);
            if ~isempty(clip_data.wearing_label)
                wearing_labels{accIndex} = clip_data.wearing_label{accIndex};
            else
                wearing_labels{accIndex} = 'unlabeled';
            end
        end
        
        acc_len = size(clip_data.acc{1},2);
        refl_acc = clip_data.acc{1} .* repmat(reflections(1,:)',1,acc_len);
        [X_total(1,:),x_labels] = getFeatures(refl_acc); 
        
        parfor acc_ind = 2:num_samples
            for reflection = 1:num_reflections
                acc_len = size(clip_data.acc{acc_ind},2);
                refl_acc = clip_data.acc{acc_ind} .* repmat(reflections(reflection,:)',1,acc_len);
                [X_total(acc_ind,:),~] = getFeatures(refl_acc); 
            end
        end
        
        features_data.subject = subject;
        features_data.feature_labels = x_labels;
        features_data.features = X_total;
        features_data.activity_labels = act_labels;
        features_data.wearing_labels = wearing_labels;
        features_data.identifier = identifier;
        features_data.activity_fraction = activity_fraction;
        features_data.times = clip_data.times';
        features_data.samplingT = options.resample_secs;
        features_data.clipdur = options.secs;
        features_data.clipoverlap = options.overlap;      
        features_data.recordtime = (round((1-options.overlap)*size(X_total,1) ) + 1)*options.secs;
        features_data.sessionID = clip_data.sessionID';
        
        %Add ID for C-Brace Subject
        subjectID = str2double(filename(4:5));
        features_data.subjectID = repmat(subjectID,num_samples,1);
        
        save(writefile,'features_data');
        disp(['Feature generataion complete for ' char(fileID)]);
    end
end
disp('----------------------------');
disp('FEATURE GENERATION COMPLETE.');
disp('----------------------------');
% end