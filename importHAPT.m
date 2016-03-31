%% Load Directories
clear all, close all;

cd(fileparts(which('importHAPT.m')))
currentDir = pwd;
slashdir = '/';
addpath([pwd slashdir 'sub']); %create path to helper scripts
cd ../
ARCode = pwd;
code_folder = [ARCode '/code/'];
unk_folder = [ARCode '/unknown_data/'];
HAPT_folder = [ARCode '/unknown_data/HAPT/'];
HAPT_raw_folder = [ARCode '/unknown_data/HAPT/RawData/'];
HAPT_feat_folder = [ARCode '/unknown_data/HAPT/Features/'];
addpath(unk_folder)
addpath(HAPT_folder)
addpath(HAPT_raw_folder)
addpath(HAPT_feat_folder)
addpath(code_folder)

%% Import Labels
%Import labels
filename = [HAPT_raw_folder 'labels.txt'];
delimiter = ' ';
formatSpec = '%f%f%f%f%f%[^\n\r]';
fileID = fopen(filename,'r');
dataArray = textscan(fileID, formatSpec, 'Delimiter', delimiter, 'MultipleDelimsAsOne', true,  'ReturnOnError', false);
fclose(fileID);
unk_labels = [dataArray{1:end-1}];
IDs = unique(unk_labels(:,2));
clear dataArray

%% Import Raw Data
%User input for which sujects to import
subjects = [];
proceed = 1;
fprintf('\n')
fprintf(2,'Please enter subjects to import. Type 0 when finished.\n')
while proceed > 0 
    subject_analyze = input('Subject ID to import: ');

    if (subject_analyze == 0)
        proceed = 0;
    else
        %Check if subjectID is in mat file
        if ~any(subject_analyze == IDs)
            disp('-------------------------------------------------')
            disp('Subject ID not found in HAPT data set. Try again.')
            disp('-------------------------------------------------')
        else
            subjects = [subjects; subject_analyze];
        end
    end
end
subjects = unique(subjects); %in case duplicate entry

%Import Each Subject Data
for ii = 1:length(subjects) %go through each subject
    X_all = [];
    temp = find(subjects(ii) == unk_labels(:,2));
    exp = unique(unk_labels(temp,1)); %the experiments for each subject
    
    %Convert subject ID to string
    if subjects(ii) < 10
        subj_str = ['0' num2str(subjects(ii))];
    elseif subjects(ii) > 9
        subj_str = num2str(subjects(ii));
    end
    
    %Go through each experiment per subject
    for jj = 1:length(exp)
        %Convert experiment ID to string
        if exp(jj) < 10
            exp_str = ['0' num2str(exp(jj))];
        elseif exp(jj) > 9
            exp_str = num2str(exp(jj));
        end
        
        %Import raw acc file
        file_acc = [HAPT_raw_folder 'acc_exp' exp_str '_user' subj_str '.txt'];
        formatSpec = '%f%f%f%[^\n\r]';
        fileID = fopen(file_acc,'r');
        accArray = textscan(fileID, formatSpec, 'Delimiter', delimiter, 'MultipleDelimsAsOne', true,  'ReturnOnError', false);
        accArray = [accArray{1:end-1}];
        fclose(fileID);
        disp(file_acc)
        
        %Convert g to m/s^2
        accArray = accArray.*9.81;
        
        %Swap x and y columns
        accArray(:,[2,1]) = accArray(:,[1,2]);
        
        %Remove unlabeled data
        ind = [1:length(accArray)]';
        comb = find(unk_labels(:,1) == exp(jj) & unk_labels(:,2) == subjects(ii));
        for zz = 1:length(comb)
            ind(unk_labels(comb(zz),4):unk_labels(comb(zz),5)) = 0;
        end
        i = find(ind == 0);
        ind(i) = [];
        accArray(ind,:) = [];
        
        %Resample from 50 Hz to 30 Hz
        acc = resample(accArray,30,50);
        
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
    
    file_save = [HAPT_feat_folder 'HAPT_' subj_str '.mat'];
    save(file_save,'X_all')
    disp(['File saved for user_' subj_str])
end