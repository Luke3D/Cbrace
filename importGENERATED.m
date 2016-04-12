%%
%INPUTS: Generated "fake" activities (Luca + Arun) csv files
%OUTPUT: One .mat file containing features matrix for all input csv files
% TO FINISH!!
%% Load Directories
clear all, close all;

cd(fileparts(which('importGENERATED.m')))
currentDir = pwd;
slashdir = '/';
addpath([pwd slashdir 'sub']); %create path to helper scripts
cd ../
ARCode = pwd;
code_folder = [ARCode '/code/'];
unk_folder = [ARCode '/unknown_data/'];
GEN_folder = [ARCode '/unknown_data/GENERATED/'];
GEN_raw_folder = [ARCode '/unknown_data/GENERATED/RawData/'];
GEN_feat_folder = [ARCode '/unknown_data/GENERATED/Features/'];
addpath(unk_folder)
addpath(GEN_folder)
addpath(GEN_raw_folder)
addpath(GEN_feat_folder)
addpath(code_folder)

p = gcp('nocreate');
if isempty(p)
    parpool('local')
end

%% Import Labels
%Import labels
% filename = [GEN_raw_folder 'labels.txt'];
% delimiter = ' ';
% formatSpec = '%f%f%f%f%f%[^\n\r]';
% fileID = fopen(filename,'r');
% dataArray = textscan(fileID, formatSpec, 'Delimiter', delimiter, 'MultipleDelimsAsOne', true,  'ReturnOnError', false);
% fclose(fileID);
% unk_labels = [dataArray{1:end-1}];
% IDs = unique(unk_labels(:,2));
% clear dataArray

%% Import Raw Data
list = dir(GEN_raw_folder);

%Import Each Subject Data
for ii = 3:6 %go through each subject
    X_all = [];
    
    %Import raw acc file
    startRow = 2;
    endRow = inf;
    delimiter = ',';
    file_acc = [GEN_raw_folder list(ii).name];
    formatSpec = '%*s%f%f%f%[^\n\r]';
    fileID = fopen(file_acc,'r');
    dataArray = textscan(fileID, formatSpec, endRow(1)-startRow(1)+1, 'Delimiter', delimiter, 'EmptyValue' ,NaN,'HeaderLines', startRow(1)-1, 'ReturnOnError', false);
    for block=2:length(startRow)
        frewind(fileID);
        dataArrayBlock = textscan(fileID, formatSpec, endRow(block)-startRow(block)+1, 'Delimiter', delimiter, 'EmptyValue' ,NaN,'HeaderLines', startRow(block)-1, 'ReturnOnError', false);
        for col=1:length(dataArray)
            dataArray{col} = [dataArray{col};dataArrayBlock{col}];
        end
    end
    fclose(fileID);
    accArray = [dataArray{1:end-1}];
    disp(file_acc)
    
    %Convert g to m/s^2
    accArray = accArray.*9.81;
    
    %         %Swap x and y columns
    %         accArray(:,[2,1]) = accArray(:,[1,2]);
    
    %Create clips
    Clipdur = 6;
    Sf = 30;
    Nfeat = 131; %number of features
    dur = Clipdur*Sf; %the clip length [samples]
    Nclips = floor(size(accArray,1)/dur);
    X_feat = zeros(Nclips,Nfeat);    %matrix with features for all clips
    parfor c = 1:Nclips
        X_feat(c,:) = getFeaturesHOME((9.81.*(acc_act_2((c-1)*dur+1:c*dur,:)))'); %generate features
    end
    
    X_all = [X_all; X_feat];    
    
    file_save = [GEN_feat_folder 'GENERATED.mat'];
    save(file_save,'X_all')
    disp(['File saved.'])
end