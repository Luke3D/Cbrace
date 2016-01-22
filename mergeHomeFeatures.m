%MERGE TWO MAT FILES FOR HOME FEATURES FILES
%EXPORT ONE COMBINED MAT FILE

%% LOAD NECESSARY DIRECTORIES
clear all; close all;
warning('off','all')
slashdir = '/';

cd(fileparts(which('mergeHomeFeatures.m')))
currentDir = pwd;
addpath([pwd slashdir 'sub']); %create path to helper scripts

current = pwd;
ARCode = current(1:end-4);
home_data_folder = [ARCode 'home_data/'];
unmerged_folder = [home_data_folder 'Unmerged/'];
addpath(home_data_folder);
addpath(unmerged_folder);
disp('Unmerged folder loaded.');

%Files in Unmerged folder
dirList = dir(unmerged_folder);
remove_ind = [];
present_IDs = zeros(length(dirList),1);
present_braces = cell(length(dirList),1);
for z = 1:length(dirList)
    dirFiles = dirList(z).name;
    if dirFiles(1) == '.' %skip hidden files
        remove_ind = [remove_ind z];
        continue
    else
        present_IDs(z) = str2num(dirFiles(4:5)); %get subject ID
        present_braces(z) = {dirFiles(7:9)}; %get corresponding brace
    end
end
dirList(remove_ind) = []; %delete hidden files
present_IDs(remove_ind) = [];
present_braces(remove_ind) = [];

%% USER INPUT FOR FILES
%Choose subject ID home data to open
proceed = 1;
while proceed > 0 
    subject_analyze = input('Enter subject ID to analyze: ');

    %Check if subjectID is in mat file
    if ~any(subject_analyze == present_IDs)
        disp('-------------------------------------------------------------')
        disp('Subject ID files not found in home_data/Unmerged/. Try again.')
        disp('-------------------------------------------------------------')
    else
        subject_indices = find(subject_analyze==present_IDs);
        subject_indices_rem = find(~subject_analyze==present_IDs);
        proceed = 0;
    end
end
present_IDs(subject_indices_rem) = [];
present_braces(subject_indices_rem) = [];

%Choose with brace to analyze
proceed = 1;
while proceed > 0
    brace_analyze = input('Brace to analyze (SCO or CBR): ','s');

    %Check if brace entered is SCO or CBR or both
    if ~(strcmpi(brace_analyze,'SCO') || strcmpi(brace_analyze,'CBR'))
        disp('--------------------------------------------------------')
        disp('Please correctly select a brace (SCO or CBR). Try again.');
        disp('--------------------------------------------------------')
    else
        if (strcmpi(brace_analyze,'CBR'))
            brace_analyze = 'CBR';

            if isempty(strmatch('CBR',present_braces))
                disp('------------------------------------------------')
                disp('CBR not found in home_data/Unmerged/. Try again.')
                disp('------------------------------------------------')
            else
                proceed = 0;
            end
        elseif (strcmpi(brace_analyze,'SCO'))
            brace_analyze = 'SCO';

            if isempty(strmatch('SCO',present_braces))
                disp('------------------------------------------------')
                disp('CBR not found in home_data/Unmerged/. Try again.')
                disp('------------------------------------------------')
            else
                proceed = 0;
            end
        end
    end
end
brace_indices = strmatch(brace_analyze,present_braces);

%% COMBINE MAT FILES
%Load mat files
fprintf('\n')
ind = subject_indices(brace_indices);
data = cell(8,length(ind)); %contains all data from all files to merge
for ii = 1:length(ind)
    disp(['Opening mat file: ' dirList(ind(ii)).name])
    load(dirList(ind(ii)).name)
    
    data{1,ii} = cut;
    data{2,ii} = data_removed;
    data{3,ii} = data_removed_str;
    data{4,ii} = days;
    data{5,ii} = size_raw;
    data{6,ii} = size_total;
    data{7,ii} = time_clips;
    data{8,ii} = Xall;
end
clear cut data_removed data_removed_str days size_raw size_total time_clips Xall

%Merge mat files
cut = 1; data_removed = 0; days = {}; size_raw = 0; size_total = 0;
time_clips = {}; Xall = [];
for kk = 1:length(ind)
    days = [days; data{4,kk}];
    size_raw = size_raw + data{5,kk};
    size_total = size_total + data{6,kk};
    time_clips = [time_clips; data{7,kk}(1:length(data{8,kk}))];
    Xall = [Xall; data{8,kk}];
end
data_removed = (1 - size_total./size_raw);
data_removed_str = num2str(data_removed.*100);
disp('Files are merged.')

%% SAVE COMBINED FILE
%Convert subject ID to string (e.g. 8 to '08')
if subject_analyze < 10
    subj_str = ['0' num2str(subject_analyze)];
elseif subject_analyze > 10
    subj_str = num2str(subject_analyze);
end

%Assemble filename
filename = ['CBR' subj_str '_' brace_analyze '_FEAT_WEAR.mat'];
cd([home_data_folder 'CBR' subj_str])
save(filename,'Xall','time_clips','days','size_total','size_raw','cut','data_removed','data_removed_str')
disp(['Merged mat file saved to home_data/CBR' subj_str '/'])