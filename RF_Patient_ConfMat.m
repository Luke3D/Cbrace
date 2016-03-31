%Averages Confusion Matrices Across Patients

%% Directories
clear; close all;
cd(fileparts(which('RF_Patient_ConfMat.m')))
currentDir = pwd;
slashdir = '/';

ARCode = currentDir(1:end-4);
conf_folder = [ARCode 'confusion_matrices/Shuffle/'];
addpath(conf_folder);

%% Import Data + Process Matrices
subj = [1 2 5 6 8 11 12 13 15 16];
%subj = [51:61];

i_total = zeros(5);
c_total = zeros(5);
matRF = zeros(5);

for ii = 1:length(subj)
    if subj(ii) < 10
        subj_str = ['0' num2str(subj(ii))];
    elseif subj(ii) > 9
        subj_str = num2str(subj(ii));
    end
    
    load([conf_folder 'CBR' subj_str '.mat']);
    i_total = i_total + instances;
    c_total = c_total + correct;
    %matRF = matRF + matRF_avg;
end 

%mat_acc = matRF./length(subj); %average accuracies across number of patients
mat_avg = i_total./c_total; %calculate raw
figure; imagesc(mat_avg); colorbar