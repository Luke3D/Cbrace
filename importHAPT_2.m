%% Load Directories
clear all, close all;

cd(fileparts(which('importHAPT_2.m')))
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

p = gcp('nocreate');
if isempty(p)
    parpool('local')
end

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
    X_labels = [];
    temp = find(subjects(ii) == unk_labels(:,2));
    labels = unk_labels(temp,:);
    exp = unique(labels(:,1)); %the unique experiments for each subject
    
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
        
        %Screen for labels only applying to specific subject AND experiment
        temp_2 = find(exp(jj) == labels(:,1));
        labels_2 = labels(temp_2,:);
        
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
        
        %Go through each label for this specific user + experiment
        for t = 1:length(labels_2)
            if labels_2(t,3) < 6 %only include 5 C-Brace activities
                acc_act = accArray(labels_2(t,4):labels_2(t,5),:); %get acc for only this specific labeled row
                acc_act_2 = resample(acc_act,30,50); %resample to 30 Hz
                
                %Convert labels to C-Brace StateCodes
                tt = labels_2(t,3);
                if tt == 1
                    a = 5; 
                elseif tt == 2
                    a = 3;
                elseif tt == 3
                    a = 2; 
                elseif tt == 4
                    a = 1; 
                elseif tt == 5
                    a = 4; 
                end
                            
                %Create clips
                Clipdur = 6;
                Sf = 30;
                Nfeat = 131; %number of features
                dur = Clipdur*Sf; %the clip length [samples]
                Nclips = floor(size(acc_act_2,1)/dur);
                X_feat = zeros(Nclips,Nfeat);    %matrix with features for all clips
                parfor c = 1:Nclips
                    X_feat(c,:) = getFeaturesHOME((9.81.*(acc_act_2((c-1)*dur+1:c*dur,:)))'); %generate features
                end
                
                X_all = [X_all; X_feat];
                X_labels = [X_labels; repmat(a,Nclips,1)];
            end    
        end
    end
    
    file_save = [HAPT_feat_folder 'HAPT_' subj_str '.mat'];
    save(file_save,'X_all', 'X_labels')
    disp(['File saved for user_' subj_str])
end