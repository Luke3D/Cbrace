%% Load Directories
clear all, close all;

cd(fileparts(which('importDSA.m')))
currentDir = pwd;
slashdir = '/';
addpath([pwd slashdir 'sub']); %create path to helper scripts
cd ../
ARCode = pwd;
code_folder = [ARCode '/code/'];
unk_folder = [ARCode '/unknown_data/'];
DSA_folder = [ARCode '/unknown_data/DSA/'];
DSA_raw_folder = [ARCode '/unknown_data/DSA/RawData/'];
DSA_feat_folder = [ARCode '/unknown_data/DSA/Features/'];
addpath(unk_folder)
addpath(DSA_raw_folder)
addpath(code_folder)

%% Import Raw Data
act_ID = [1:19]; %activities in DSA set to analyze

%Import Each Activity Folder
for ii = 1:length(act_ID) %go through each activity
    X_all = []; %stores all activity data
    
    %Convert activity ID to string
    if act_ID(ii) < 10
        act_str = ['0' num2str(act_ID(ii))];
    elseif act_ID(ii) > 9
        act_str = num2str(act_ID(ii));
    end
    
    %Activity folder
    act_folder = [DSA_raw_folder 'a' act_str '/'];
    disp(act_folder)
    
    %Create dir struct + remove temp files
    dirList1 = dir(act_folder);
    remove_ind = [];
    for z = 1:length(dirList1)
        dirFiles = dirList1(z).name;
        if dirFiles(1) == '.' %skip hidden files
            remove_ind = [remove_ind z];            
        end
    end
    dirList1(remove_ind) = []; %delete hidden files
    
    %Go through each folder within activity folder
    for jj = 1:length(dirList1)
        subj_folder = [act_folder dirList1(jj).name '/'];
        
        %Create dir struct + remove temp files
        dirList2 = dir(subj_folder);
        remove_ind = [];
        for zz = 1:length(dirList1)
            dirFiles = dirList2(zz).name;
            if dirFiles(1) == '.' %skip hidden files
                remove_ind = [remove_ind zz];
            end
        end
        dirList2(remove_ind) = []; %delete hidden files
        
        accArray = [];
        %Import and concatenate all .txt files
        for tt = 1:length(dirList2)
            file_acc = [subj_folder dirList2(tt).name];
            delimiter = ',';
            formatSpec = '%f%f%f%*s%*s%*s%*s%*s%*s%*s%*s%*s%*s%*s%*s%*s%*s%*s%*s%*s%*s%*s%*s%*s%*s%*s%*s%*s%*s%*s%*s%*s%*s%*s%*s%*s%*s%*s%*s%*s%*s%*s%*s%*s%*s%[^\n\r]';
            fileID = fopen(file_acc,'r');
            data = textscan(fileID, formatSpec, 'Delimiter', delimiter,  'ReturnOnError', false);
            data = [data{1:end-1}];
            fclose(fileID);

            accArray = [accArray; data]; %Concatenate data
        end
        
        %Swap x and y columns
        accArray(:,[2,1]) = accArray(:,[1,2]);
        
        %Resample from 25 Hz to 30 Hz
        acc = resample(accArray,30,25);
        
        %Generate features
        p = gcp('nocreate');
        if isempty(p)
            parpool('local')
        end
        Clipdur = 6;
        Sf = 30;
        Nfeat = 131; %number of features
        dur = Clipdur*Sf; %the clip length [samples]
        Nclips = floor(size(acc,1)/dur);
        X_feat = zeros(Nclips,Nfeat);    %matrix with features for all clips
        parfor c = 1:Nclips
            X_feat(c,:) = getFeaturesHOME((9.81.*(acc((c-1)*dur+1:c*dur,:)))'); %generate fatures
        end
        %Concatenate features
        X_all = [X_all; X_feat];
    end
        
    file_save = [DSA_feat_folder 'DSA_' act_str '.mat'];
    save(file_save,'X_all')
    disp(['File saved for activity_' act_str])
    fprintf('\n')
end