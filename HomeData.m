%TRAIN RF AND HMM ON A COMPLETE DATASET AND TEST ON HOME DATA

%% LOAD NECESSARY DIRECTORIES
clear all; close all;
warning('off','all')
slashdir = '/';

cd(fileparts(which('HomeData.m')))
currentDir = pwd;
addpath([pwd slashdir 'sub']); %create path to helper scripts
addpath(genpath([slashdir 'Traindata'])); %add path for train data

current = pwd;
ARCode = current(1:end-4);
home_data_folder = [ARCode 'home_data/'];
addpath(home_data_folder);
disp('Home Data folder loaded.');

ExtractAllFeatures = 0;                 %if all features need to be extracted
ExtractWearFeatures = 0;                %if all features need to be extracted
WearTime = 1;                           %if Wear time only are extracted
plotON = 1;                             %draw plots
drawplot.activities = 0;
drawplot.actvstime = 1;
drawplot.confmat = 1;
UseHMM = 1;  
subject_stairs = [2 8]; %List of subject IDs for stairs to remove

%% ASK TO USE LAB/HOME/BOTH LABELED DATA FOR TRAINING
use_home = 0; use_lab = 0;
proceed = 1;
fprintf(2,'FOR TRAINING DATA:\n');
while proceed > 0
    train = input('Train on labeled data from lab or home or both? ','s');
    if strcmpi(train,'lab')
        use_lab = 1;
        proceed = 0;
    elseif strcmpi(train,'home')
        use_home = 1;
        proceed = 0;
    elseif strcmpi(train,'both')
        use_home = 1;
        use_lab = 1;
        proceed = 0;
    else
        disp('Please type lab or home or both.');
        proceed = 1;
    end
end

%% LOAD LAB TRAINING DATA + SELECT SUBJECT/BRACE TO ANALYZE
if use_lab
    %Load .mat file with all patient lab training data
    fprintf('\n')
    filename = 'trainData_patient.mat';
    load(filename)
    fprintf(2,'FOR THE LAB TRAINING DATA:\n')
    disp('Lab labeled training data file loaded.')

    tt = num2str(unique(trainingClassifierData.subjectID)');
    fprintf('\n')
    fprintf('Subject IDs present for analysis: %s',tt)
    fprintf('\n')
    fprintf('Available files to analyze: ')
    fprintf('\n')
    disp(unique(trainingClassifierData.subject))
    fprintf('\n')

    all_subjectID = trainingClassifierData.subjectID;

    %Choose subject ID lab data to open
    proceed = 1;
    while proceed > 0 
        subject_analyze = input('Enter subject ID to analyze: ');

        %Check if subjectID is in mat file
        if ~any(subject_analyze == all_subjectID)
            disp('-------------------------------------------------------------')
            disp('Subject ID not in trainingClassifierData.mat file. Try again.')
            disp('-------------------------------------------------------------')
        else
            subject_indices = find(subject_analyze==all_subjectID);
            proceed = 0;
        end
    end

    %Remove all other subjects
    cData_temp2 = isolateSubject(trainingClassifierData,subject_indices);

    %Set up directories for this subject
    if subject_analyze < 10
        subj_str = ['0' num2str(subject_analyze)];
    elseif subject_analyze > 10
        subj_str = num2str(subject_analyze);
    end

    for zz = 1:length(cData_temp2.subject)
        temp = char(cData_temp2.subject(zz));
        cData_temp2.subjectBrace(zz) = {temp(7:9)};
    end

    %Choose with brace to analyze
    proceed = 1;
    while proceed > 0
        fprintf('\n')
        brace_analyze = input('Brace to analyze (SCO, CBR, both): ','s');

        %Check if brace entered is SCO or CBR or both
        if ~(strcmpi(brace_analyze,'SCO') || strcmpi(brace_analyze,'CBR') || strcmpi(brace_analyze,'BOTH'))
            disp('---------------------------------------------------------------')
            disp('Please correctly select a brace (SCO, CBR, or both). Try again.');
            disp('---------------------------------------------------------------')
        else
            %Check if SCO or CBR are in mat file
            if (strcmpi(brace_analyze,'both'))
                brace_analyze = 'both';

                if isempty(strmatch('Cbr',cData_temp2.subjectBrace)) || isempty(strmatch('SCO',cData_temp2.subjectBrace))
                    disp('--------------------------------------------------------')
                    disp('Brace not in trainingClassifierData.mat file. Try again.')
                    disp('--------------------------------------------------------')
                else
                    proceed = 0;
                end
            elseif (strcmpi(brace_analyze,'CBR'))
                brace_analyze = 'Cbr';

                if isempty(strmatch('Cbr',cData_temp2.subjectBrace))
                    disp('------------------------------------------------------')
                    disp('CBR not in trainingClassifierData.mat file. Try again.')
                    disp('------------------------------------------------------')
                else
                    proceed = 0;
                end
            elseif (strcmpi(brace_analyze,'SCO'))
                brace_analyze = 'SCO';

                if isempty(strmatch('SCO',cData_temp2.subjectBrace))
                    disp('------------------------------------------------------')
                    disp('SCO not in trainingClassifierData.mat file. Try again.')
                    disp('------------------------------------------------------')
                else
                    proceed = 0;
                end
            end
        end
    end

    %Remove all other braces
    cData_temp = isolateBrace(cData_temp2,brace_analyze);

    %Choose lab sessions to analyze
    fprintf('\n')
    disp('Enter session IDs to analyze or type 0 to analyze all.')
    disp('Press ENTER when you finish inputting numbers.')
    a = 0; 
    if (strcmpi(brace_analyze,'both'))
        IDs = [];
        disp('Lab sessions for Cbr brace:');
        while a == 0
            i = input('Session ID: ');
            if i == 0
                a = 1;
                disp('Using all lab sessions for this brace.')
            elseif isempty(i)
                a = 1;
            else
               IDs = [IDs; i]; 
            end
        end
        if ~isempty(IDs) 
            cData = isolateSessionSpec(cData_temp,IDs,'Cbr');
        else
            cData = cData_temp;    
        end

        a = 0; IDs = [];
        disp('Lab sessions for SCO brace:');
        while a == 0
            i = input('Session ID: ');
            if i == 0
                a = 1;
                disp('Using all lab sessions for this brace.')
            elseif isempty(i)
                a = 1;
            else
               IDs = [IDs; i]; 
            end
        end
        if ~isempty(IDs) 
            cData = isolateSessionSpec(cData,IDs,'Cbr');
        else
            cData = cData_temp;    
        end
    elseif (strcmpi(brace_analyze,'CBR')) || (strcmpi(brace_analyze,'SCO'))
        IDs = [];
        while a == 0
            i = input('Session ID: ');
            if i == 0
                a = 1;
                disp('Using all lab sessions for this brace.')
            elseif isempty(i)
                a = 1;
            else
               IDs = [IDs; i]; 
            end
        end    
        if ~isempty(IDs) 
            cData = isolateSessionSpec(cData_temp,IDs,'Cbr');
        else
            cData = cData_temp;
        end
    end

    %Display which training file will be used
    fprintf('\n')
    disp('These are the subject files that will be analyzed: ')
    disp(unique(cData.subject))
    
    cData_lab = cData;
else
    cData_lab.activity = [];
    cData_lab.wearing = [];
    cData_lab.identifier = [];
    cData_lab.subject = [];
    cData_lab.features = [];
    cData_lab.activityFrac = [];
    cData_lab.subjectID = [];
    cData_lab.sessionID = [];
    cData_lab.states = [];
end

%% LOAD HOME TRAINING DATA + SELECT SUBJECT/BRACE TO ANALYZE
if use_home
    %Load .mat file with all patient lab training data
    fprintf('\n')
    filename = 'trainData_home.mat';
    load(filename)
    fprintf(2,'FOR THE HOME TRAINING DATA:\n')
    disp('Home labeled training data file loaded.')

    tt = num2str(unique(trainingClassifierData.subjectID)');
    fprintf('\n')
    fprintf('Subject IDs present for analysis: %s',tt)
    fprintf('\n')
    fprintf('Available files to analyze: ')
    fprintf('\n')
    disp(unique(trainingClassifierData.subject))
    fprintf('\n')

    all_subjectID = trainingClassifierData.subjectID;

    %Choose subject ID home data to open
    proceed = 1;
    while proceed > 0 
        subject_analyze = input('Enter subject ID to analyze: ');

        %Check if subjectID is in mat file
        if ~any(subject_analyze == all_subjectID)
            disp('-------------------------------------------------------------')
            disp('Subject ID not in trainingClassifierData.mat file. Try again.')
            disp('-------------------------------------------------------------')
        else
            subject_indices = find(subject_analyze==all_subjectID);
            proceed = 0;
        end
    end

    %Remove all other subjects
    cData_temp2 = isolateSubject(trainingClassifierData,subject_indices);

    %Set up directories for this subject
    if subject_analyze < 10
        subj_str = ['0' num2str(subject_analyze)];
    elseif subject_analyze > 10
        subj_str = num2str(subject_analyze);
    end
    

    for zz = 1:length(cData_temp2.subject)
        temp = char(cData_temp2.subject(zz));
        cData_temp2.subjectBrace(zz) = {temp(7:9)};
    end

    %Choose with brace to analyze
    proceed = 1;
    while proceed > 0
        fprintf('\n')
        brace_analyze = input('Brace to analyze (SCO, CBR, both): ','s');

        %Check if brace entered is SCO or CBR or both
        if ~(strcmpi(brace_analyze,'SCO') || strcmpi(brace_analyze,'CBR') || strcmpi(brace_analyze,'BOTH'))
            disp('---------------------------------------------------------------')
            disp('Please correctly select a brace (SCO, CBR, or both). Try again.');
            disp('---------------------------------------------------------------')
        else
            %Check if SCO or CBR are in mat file
            if (strcmpi(brace_analyze,'both'))
                brace_analyze = 'both';

                if isempty(strmatch('Cbr',cData_temp2.subjectBrace)) || isempty(strmatch('SCO',cData_temp2.subjectBrace))
                    disp('--------------------------------------------------------')
                    disp('Brace not in trainingClassifierData.mat file. Try again.')
                    disp('--------------------------------------------------------')
                else
                    proceed = 0;
                end
            elseif (strcmpi(brace_analyze,'CBR'))
                brace_analyze = 'Cbr';

                if isempty(strmatch('Cbr',cData_temp2.subjectBrace))
                    disp('------------------------------------------------------')
                    disp('CBR not in trainingClassifierData.mat file. Try again.')
                    disp('------------------------------------------------------')
                else
                    proceed = 0;
                end
            elseif (strcmpi(brace_analyze,'SCO'))
                brace_analyze = 'SCO';

                if isempty(strmatch('SCO',cData_temp2.subjectBrace))
                    disp('------------------------------------------------------')
                    disp('SCO not in trainingClassifierData.mat file. Try again.')
                    disp('------------------------------------------------------')
                else
                    proceed = 0;
                end
            end
        end
    end

    %Remove all other braces
    cData_temp = isolateBrace(cData_temp2,brace_analyze);

    %Choose home sessions to analyze
    fprintf('\n')
    disp('Enter session IDs to analyze or type 0 to analyze all.')
    disp('Press ENTER when you finish inputting numbers.')
    a = 0; 
    if (strcmpi(brace_analyze,'both'))
        IDs = [];
        disp('Home sessions for Cbr brace:');
        while a == 0
            i = input('Session ID: ');
            if i == 0
                a = 1;
                disp('Using all home sessions for this brace.')
            elseif isempty(i)
                a = 1;
            else
               IDs = [IDs; i]; 
            end
        end
        if ~isempty(IDs) 
            cData = isolateSessionSpec(cData_temp,IDs,'Cbr');
        else
            cData = cData_temp;    
        end

        a = 0; IDs = [];
        disp('Home sessions for SCO brace:');
        while a == 0
            i = input('Session ID: ');
            if i == 0
                a = 1;
                disp('Using all home sessions for this brace.')
            elseif isempty(i)
                a = 1;
            else
               IDs = [IDs; i]; 
            end
        end
        if ~isempty(IDs) 
            cData = isolateSessionSpec(cData,IDs,'Cbr');
        else
            cData = cData_temp;    
        end
    elseif (strcmpi(brace_analyze,'CBR')) || (strcmpi(brace_analyze,'SCO'))
        IDs = [];
        while a == 0
            i = input('Session ID: ');
            if i == 0
                a = 1;
                disp('Using all home sessions for this brace.')
            elseif isempty(i)
                a = 1;
            else
               IDs = [IDs; i]; 
            end
        end    
        if ~isempty(IDs) 
            cData = isolateSessionSpec(cData_temp,IDs,'Cbr');
        else
            cData = cData_temp;
        end
    end

    %Display which training file will be used
    fprintf('\n')
    disp('These are the subject files that will be analyzed: ')
    disp(unique(cData.subject))
    
    cData_home = cData;
else
    cData_home.activity = [];
    cData_home.wearing = [];
    cData_home.identifier = [];
    cData_home.subject = [];
    cData_home.features = [];
    cData_home.activityFrac = [];
    cData_home.subjectID = [];
    cData_home.sessionID = [];
    cData_home.states = [];
end

%% DETERMINE ALL VS. WEAR FOR HOME DATA
fprintf('\n')
fprintf(2,'FOR TESTING HOME DATA:\n')

%Ask for which subject to use
subject_analyze = input('Enter subject ID to analyze: ');
if subject_analyze < 10
    subj_str = ['0' num2str(subject_analyze)];
elseif subject_analyze > 10
    subj_str = num2str(subject_analyze);
end

%Directory information
home_folder = [ARCode 'home_data/CBR' subj_str '/'];
addpath(home_folder);
disp('Subject home data folder loaded.')
dirList = dir(home_folder);

%Ask for which brace to use
proceed = 1;
while proceed > 0
    fprintf('\n')
    brace_analyze = input('Brace to analyze (SCO or CBR): ','s');
    if (strcmpi(brace_analyze,'CBR'))
        brace_analyze = 'Cbr';
        proceed = 0;
    elseif (strcmpi(brace_analyze,'SCO'))
        brace_analyze = 'SCO';
        proceed = 0;
    else
        disp('Please select SCO or CBR');
        proceed = 1;
    end
end

%Ask for whether to use all or only wear data
proceed = 1;
while proceed > 0
    fprintf('\n')
    feature_type = input('Use only wear data (yes/no): ','s');
    if strcmpi(feature_type,'yes') %use only wear information
        fprintf('\n')
        disp('Only wear data will be used.');
        file_end = '_FEAT_WEAR';
        proceed = 0;
    elseif strcmpi(feature_type,'no') %use all information
        fprintf('\n')
        disp('All data will be used.');
        file_end = '_FEAT_ALL';
        proceed = 0;
    else
        disp('Please type yes or no.')
    end
end

%Check if all/wear features file is present
exists = 0;
for z = 1:length(dirList)
    dirFiles = dirList(z).name;
    if dirFiles(1) == '.' %skip hidden files
        continue
    else
        dirFiles = dirFiles(1:end-4); %remove .csv from string
        if strcmpi(dirFiles(4:5),subj_str) && strcmpi(dirFiles(7:end),[brace_analyze file_end])
            disp([file_end(7:end) ' features file present for CBR' subj_str ' - ' upper(brace_analyze) '.'])
            exists = 1;
        end
    end
end

if ~exists %features do not already exist
    disp([file_end(7:end) ' features file is NOT present for CBR' subj_str ' - ' upper(brace_analyze) '.'])
    disp([file_end(7:end) ' features file will be generated.'])
    
    if strcmpi(feature_type,'yes') %use only wear information
        ExtractWearFeatures = 1;
    elseif strcmpi(feature_type,'no') %use all information
        ExtractAllFeatures = 1;
    end
end

%% HOME DATA DIRECTORIES
filename = '';
if ExtractWearFeatures || ExtractAllFeatures
    for directory = 1:length(dirList)
        dirName = dirList(directory).name;
        if dirName(1) == '.' %skip hidden files
            continue
        else
            dirName = dirName(1:end-4); %remove .csv from string
            if strcmpi(dirName(23:end),brace_analyze) %check end of file name
                disp([brace_analyze ' home data located for CBR' subj_str '.'])
                filename = dirName;
            end
        end
    end

    if isempty(filename)
        error('Home data file for this brace could not be located.');
    else
        datafile = [home_folder filename];
    end
end

%% Variable Initialization
if ExtractAllFeatures || ExtractWearFeatures
    if use_home
        Clipdur = cData_home.clipdur;                  %clip length in secs
        Nfeat = size(cData_home.features,2);       %number of features
    else
        Clipdur = cData_lab.clipdur;                  %clip length in secs
        Nfeat = size(cData_lab.features,2);       %number of features
    end
    Sf = 30;                                  %the sampling freq [Hz]
    dur = Clipdur*Sf;    %the clip length [samples]
    size_total = 0; size_raw = 0;
    startFound = 0; endFound = 0;
    days = {}; time_clips = {};
    Xall = [];
end

%% LOAD ALL DATA + GENERATE FEATURES (only if ALL features option is selected)
if ExtractAllFeatures
    p = gcp('nocreate');
    if isempty(p)
        parpool('local')
    end
    
    tic
    %Load home data accelerations
    ds = datastore([datafile '.csv'],'NumHeaderLines',10,'ReadVariableNames',1,'SelectedVariableNames',{'Timestamp','AccelerometerX','AccelerometerY','AccelerometerZ'});
    ds.RowsPerRead = 360000; %integer number of clips
    while hasdata(ds)
        imp = read(ds);
        timestamp = imp.Timestamp;
        acc = [imp.AccelerometerX imp.AccelerometerY imp.AccelerometerZ];
        size_raw = size_raw + length(acc);
        cut = 1;
        k1 = cell2mat(strfind(timestamp,' '));
        td = cellfun(@(s,k) s(1:k),timestamp,num2cell(k1),'uni',0);
        [~,yy] = unique(td,'first');
        days = [days; td(sort(yy))]; %prevent unique from auto sorting dates
        
        %Extract clips and generate features for clips
        Nclips = floor(size(acc,1)/dur);
        X_feat = zeros(Nclips,Nfeat);    %matrix with features for all clips
        parfor c = 1:Nclips
            X_feat(c,:) = getFeaturesHOME((9.81.*(acc((c-1)*dur+1:c*dur,:)))'); %generate fatures
        end
        Xall = [Xall; X_feat];
        time_clips = [time_clips; timestamp(1:dur:end)];
    end
    
    %Remove duplicate days in days vector
    [~,idx] = unique(strcat(days(:,1)));
    days = days(sort(idx),:);
    
    data_removed = 0;
    data_removed_str = num2str(data_removed.*100);
    disp(['Data removed: 0%']);
    
    clearvars imp acc X_feat timestamp
            
    fprintf('\n')
    tel = toc;
    disp('Home data imported and features extracted.')
    disp([num2str(tel) ' seconds elapsed.'])
    
    %Save features file
    fprintf('\n')
    home_folder = [ARCode 'home_data/CBR' subj_str '/'];
    cd(home_folder)
    disp(['Saving ' file_end(7:end) ' features file...'])
    filename = [filename(1:5) '_' upper(brace_analyze) file_end '.mat'];
    save(filename,'Xall','time_clips','days','size_total','size_raw','cut','data_removed','data_removed_str')
    disp('File saved.')
    fprintf('\n')    
end

%% REMOVE NON-WEARING DATA + GENERATE FEATURES (only if WEAR features option is selected)
if ExtractWearFeatures
    p = gcp('nocreate');
    if isempty(p)
        parpool('local')
    end
    
    %Load home data accelerations
    ds = datastore([datafile '.csv'],'NumHeaderLines',10,'ReadVariableNames',1,'SelectedVariableNames',{'Timestamp','AccelerometerX','AccelerometerY','AccelerometerZ'});
    ds.RowsPerRead = 360000; %integer number of clips
    
    %Load non-wearing data (determined by ActiLife)
    ds_wear = datastore([datafile '_Wearing.csv'],'ReadVariableNames',1,'SelectedVariableNames',{'Date/Time Start','Date/Time Stop','Wear or Non-Wear'});
    ds_wear.RowsPerRead = 360000;
    if hasdata(ds_wear)
        wear_tbl = read(ds_wear);
        wear = table2cell(wear_tbl);
        fprintf('\n')
        disp('Wear and Non-Wear info is loaded.')
    else
        error('Wear and Non-Wear data is not located.')
    end
    
    %Remove "Wear" data and format timestamps
    nonwear_ind = strmatch('Wear',wear(:,3));
    wear(nonwear_ind,:) = [];
    N_wear = length(wear);
    start_time = cell(N_wear,1);
    stop_time = cell(N_wear,1); 
    for ii = 1:N_wear
        k1 = strfind(wear(ii,1),' ');
        k1 = cell2mat(k1); %convert string to number
        temp1 = char(wear(ii,1));
        if length(temp1(k1+1:end)) == 4
            start_time(ii) = strcat(cellstr(temp1(1:k1)),' 0',cellstr(temp1(k1+1:end)),':00.000');
        else
            start_time(ii) = strcat(wear(ii,1),':00.000');
        end
        
        k2 = strfind(wear(ii,2),' ');
        k2 = cell2mat(k2); %convert string to number
        temp2 = char(wear(ii,2));
        if length(temp2(k2+1:end)) == 4
            stop_time(ii) = strcat(cellstr(temp2(1:k2)),' 0',cellstr(temp2(k2+1:end)),':00.000');
        else
            stop_time(ii) = strcat(wear(ii,2),':00.000');
        end
    end
    disp('Wear data removed and timestamps formatted.')
    
    tic
    %Cycle through non-wear timestamp pairs
    disp('Isolating wear data...')
    readchunk = 1;
    k = 1;
    chk_counter = 0;
    increment = 0;
    fprintf('\n')
    disp(['Timestamp: ' num2str(k) ' of ' num2str(N_wear)]);
    while k < (N_wear+1) %for each time stamp pair
        
        %readchunk = 0: stay on current chunk (next timestamp)
        %readchunk = 1: go to next chunk (same timestamp)
        %readchunk = 2: go to next chunk + start found in previous chunk
        %Read new chunk
        if readchunk > 0
            imp = read(ds);
            timestamp = imp.Timestamp;
            acc = [imp.AccelerometerX imp.AccelerometerY imp.AccelerometerZ];
            size_raw = size_raw + length(acc);
            cut = 1;            
            chk_counter = chk_counter + 1;
            k1 = cell2mat(strfind(timestamp,' '));
            td = cellfun(@(s,k) s(1:k),timestamp,num2cell(k1),'uni',0);
            [~,yy] = unique(td,'first');
            days = [days; td(sort(yy))]; %prevent unique from auto sorting dates
        end
        
        if increment
            fprintf('\n')
            disp(['Timestamp: ' num2str(k) ' of ' num2str(N_wear)]);
            increment = 0;            
        end
        
        fprintf(['Chunk: ' num2str(chk_counter) ' | ']);

        %Look if start and stop timestamps are present (true/false)
        startFound = ~isempty(strmatch(start_time(k),timestamp,'exact'));
        stopFound = ~isempty(strmatch(stop_time(k),timestamp,'exact'));

        if startFound && stopFound
            start_ind = strmatch(start_time(k),timestamp,'exact');
            stop_ind = strmatch(stop_time(k),timestamp,'exact');

            %Remove non-wearing data
            acc(start_ind:stop_ind,:) = [];
            timestamp(start_ind:stop_ind,:) = [];

            %Go to next timestamp and do not read next chunk
            readchunk = 0; 
            k = k + 1; increment = 1;
            disp('Start/stop found. Removed start to stop.')
        end

        if startFound && ~stopFound
            start_ind = strmatch(start_time(k),timestamp,'exact');

            %Remove rest of data in chunk
            acc(start_ind:end,:) = [];
            timestamp(start_ind:end,:) = [];

            %Do not go to next timestamp and read next chunk
            readchunk = 2;
            k = k;
            disp('Start found. Remove start till end.')
        end

        if ~startFound && stopFound
            stop_ind = strmatch(stop_time(k),timestamp,'exact');

            %Remove data till end timestamp
            acc(1:stop_ind,:) = [];
            timestamp(1:stop_ind,:) = [];

            %Go to next timestamp and do not read next chunk
            readchunk = 0; 
            k = k + 1; increment = 1;
            disp('Stop found. Remove beginning till stop.')
        end

        if ~startFound && ~stopFound
            %Reset readchunk
            if readchunk == 0
                readchunk = 1;
                disp('Start/stop not found. Next chunk.')
            
            %Redundant (just to explain this possibility)
            elseif readchunk == 1
                readchunk = 1;
                disp('Start/stop not found. Next chunk.')
            end

            if readchunk == 2 %start was found in previous chunk
                acc = [];
                timestamp = [];
                disp('Start/stop not found. Remove all - looking for stop. Next chunk.')
            end
            
        end
        
        if readchunk > 0 || (k-1) == N_wear            
            %Extract clips and generate features for clips
            Nclips = floor(size(acc,1)/dur);
            X_feat = zeros(Nclips,Nfeat);    %matrix with features for all clips
            parfor c = 1:Nclips
                X_feat(c,:) = getFeaturesHOME((9.81.*(acc((c-1)*dur+1:c*dur,:)))'); %generate fatures
            end
            Xall = [Xall; X_feat];
            time_clips = [time_clips; timestamp(1:dur:end)];
        end
    end
    
    fprintf('\n')
    disp('Remaining Wearing Data:');
    while hasdata(ds)
        fprintf(['Chunk: ' num2str(chk_counter) ' | Reading all.']);

        imp = read(ds);
        timestamp = imp.Timestamp;
        acc = [imp.AccelerometerX imp.AccelerometerY imp.AccelerometerZ];
        size_raw = size_raw + length(acc);
        chk_counter = chk_counter + 1;
        k1 = cell2mat(strfind(timestamp,' '));
        td = cellfun(@(s,k) s(1:k),timestamp,num2cell(k1),'uni',0);
        [~,yy] = unique(td,'first');
        days = [days; td(sort(yy))]; %prevent unique from auto sorting dates
        
        Nclips = floor(size(acc,1)/dur);
        X_feat = zeros(Nclips,Nfeat);    %matrix with features for all clips
        parfor c = 1:Nclips
            X_feat(c,:) = getFeaturesHOME((9.81.*(acc((c-1)*dur+1:c*dur,:)))'); %generate fatures
        end
        Xall = [Xall; X_feat];
        time_clips = [time_clips; timestamp(1:dur:end)];
    end
    
    %Remove duplicate days in days vector
    [~,idx] = unique(strcat(days(:,1)));
    days = days(sort(idx),:);
    
    %Display amount of non-wear data removed
    fprintf('\n');
    disp('Wearing data isolated.');
    size_total = length(Xall).*180;
    data_removed = (1 - size_total./size_raw);
    data_removed_str = num2str(data_removed.*100);
    disp(['Data removed: ' data_removed_str(1:5) '%']);
    
    clearvars imp acc X_feat timestamp
            
    fprintf('\n')
    tel = toc;
    disp('Home data imported and features extracted.')
    disp([num2str(tel) ' seconds elapsed.'])
    
    %Save features file
    fprintf('\n')
    home_folder = [ARCode 'home_data/CBR' subj_str '/'];
    cd(home_folder)
    disp(['Saving ' file_end(7:end) ' features file...'])
    filename = [filename(1:5) '_' upper(brace_analyze) file_end '.mat'];
    save(filename,'Xall','time_clips','days','size_total','size_raw','cut','data_removed','data_removed_str')
    disp('File saved.')
    fprintf('\n')
end

%% LOAD FEATURES FILE (if features file is already present)
if ~ExtractAllFeatures && ~ExtractWearFeatures
    disp('Home data features file is loading...')
    features_file = ['CBR' subj_str '_' upper(brace_analyze) file_end '.mat'];
    load(features_file)
    disp(['Home data ' file_end(7:end) ' features file is loaded.'])
    fprintf('\n')
end
    
%% PREPARE FOR TRAINING
cd([ARCode 'code/'])

%Clip threshold options
clipThresh = 0; %to be in training set, clips must have >X% of label

% cData = scaleFeatures(cData); %scale to [0 1]
%cData = cData;

%remove data from other locations if required (old datasets)
if use_lab
    cData_lab = removeDataWithoutLocation(cData_lab,'Belt');
end
if use_home
    cData_home = removeDataWithoutLocation(cData_home,'Belt');
end

%create local variables for often used data
features     = [cData_lab.features; cData_home.features]; %features for classifier
subjects     = [cData_lab.subject; cData_home.subject];  %subject number
uniqSubjects = unique(subjects); %list of subjects
statesTrue   = [cData_lab.activity; cData_home.activity];     %all the classifier data
uniqStates   = unique(statesTrue);  %set of states we have

%How many clips of each activity type we removed
if clipThresh > 0
    %remove any clips that don't meet the training set threshold
    if use_lab
        [cData_lab, removeInd] = removeDataWithActivityFraction(cData_lab,clipThresh);
    end
    if use_home
        [cData_home, removeInd] = removeDataWithActivityFraction(cData_home,clipThresh);
    end
    
    fprintf('\n')
    for i = 1:length(uniqStates)
        indr = find(strcmp(trainingClassifierData.activity(removeInd),uniqStates(i)));
        indtot = find(strcmp(trainingClassifierData.activity,uniqStates(i)));
        removed = length(indr)/length(indtot)*100;
        disp([num2str(removed) ' % of ' uniqStates{i} ' data removed'])
    end
end

%Get codes for the true states (i.e. make a number code for each state) and save code and state
codesTrue = zeros(1,length(statesTrue));
for i = 1:length(statesTrue)
    codesTrue(i) = find(strcmp(statesTrue{i},uniqStates));
end

%REMOVE STAIRS FOR SPECIFIC SUBJECTS
if ~isempty(find(subject_stairs==subject_analyze, 1))
    stairs_ind = [find(codesTrue == 2) find(codesTrue == 3)];
    if ~isempty(stairs_ind)
        %Remove stairs training data:
        features(stairs_ind,:) = [];
        statesTrue(stairs_ind) = [];
        subjects(stairs_ind) = [];
        codesTrue(stairs_ind) = [];
        uniqStates([2 3]) = []; %only three classes remaining
        
        %Update codesTrue to use 1 2 3 (not 1 4 5)
        temp = find(codesTrue == 4);
        codesTrue(temp) = 2;
        temp = find(codesTrue == 5);
        codesTrue(temp) = 3;
        
        fprintf(2,'Stairs data removed from training data.\n')
    else
        fprintf(2,'No stairs data found in training data.\n')
    end
end

%Store Code and label of each unique State
StateCodes = cell(length(uniqStates),2);
StateCodes(:,1) = uniqStates;
StateCodes(:,2) = num2cell(1:length(uniqStates)); %sorted by unique

%% TRAIN RF (standard parameters and save results)
ntrees = 100;
fprintf('\n')
disp(['RF Train - Number of samples train = ' num2str(size(features,1))])
disp(['Sitting:   ' num2str(length(strmatch('Sitting',statesTrue,'exact'))) ' (' num2str((length(strmatch('Sitting',statesTrue,'exact'))./length(statesTrue))*100) '%)']);
disp(['Stairs Dw: ' num2str(length(strmatch('Stairs Dw',statesTrue,'exact'))) ' (' num2str((length(strmatch('Stairs Dw',statesTrue,'exact'))./length(statesTrue))*100) '%)']);
disp(['Stairs Up: ' num2str(length(strmatch('Stairs Up',statesTrue,'exact'))) ' (' num2str((length(strmatch('Stairs Up',statesTrue,'exact'))./length(statesTrue))*100) '%)']);
disp(['Standing:  ' num2str(length(strmatch('Standing',statesTrue,'exact'))) ' (' num2str((length(strmatch('Standing',statesTrue,'exact'))./length(statesTrue))*100) '%)']);
disp(['Walking:   ' num2str(length(strmatch('Walking',statesTrue,'exact'))) ' (' num2str((length(strmatch('Walking',statesTrue,'exact'))./length(statesTrue))*100) '%)']);
fprintf('\n')

RFmodel = TreeBagger(ntrees,features,codesTrue');

%RF Prediction and RF class probabilities for ENTIRE dataset. This is
%for initializing the HMM Emission matrix (P_RF(TrainSet)) and for
%computing the observations of the HMM (P_RF(TestSet))
[codesRF,P_RF] = predict(RFmodel,features);
codesRF = str2num(cell2mat(codesRF));
statesRF = uniqStates(codesRF);

disp('RF Model trained.')

%% TRAIN HMM (i.e. create HMM and set the emission prob as the RF output)
PTrain = P_RF;  %The Emission Probabilities of the HMM are the RF output prob on the train dataset

%Initialize parameters for hmm
d       = length(uniqStates);   %number of symbols (=#states)
nstates = d;                    %number of states
mu      = zeros(d,nstates);     %mean of emission distribution
sigma   = zeros(d,1,nstates);   %std dev of emission distribution
Pi      = ones(length(uniqStates),1) ./ length(uniqStates); %uniform prior
sigmaC  = .1;                   %use a constant std dev

%The HMM Transition Matrix (A)
if d == 3
    transitionFile = 'A_3Activity.xlsx';
elseif d == 5
    transitionFile = 'A_5ActivityNSS.xlsx';
else
    error('Appropriate transition matrix file for HMM is not selected.')
end
A = xlsread(transitionFile);

%Create emission probabilities for HMM
PBins = cell(d,1);

%For each type of state we need a distribution
for bin = 1:d
    clipInd         = strcmp(uniqStates{bin},statesTrue);
    PBins{bin,1}    = PTrain(clipInd,:);
    mu(:,bin)       = mean(PBins{bin,1}); %mean
    sigma(:,:,bin)  = sigmaC; %set std dev
end

%Create distribution for pmtk3 package
emission        = struct('Sigma',[],'mu',[],'d',[]);
emission.Sigma  = sigma;
emission.mu     = mu;
emission.d      = d;

%Construct HMM using pmtk3 package
HMMmodel           = hmmCreate('gauss',Pi,A,emission);
HMMmodel.emission  = condGaussCpdCreate(emission.mu,emission.Sigma);
HMMmodel.fitType   = 'gauss';

disp('HMM Model trained.')

%Save RF and HMM models
disp('Saving RF and HMM models...')
file_models = ['CBR' subj_str '_' upper(brace_analyze) '_MODELS' '_' file_end(7:end) '.mat'];
code_folder = cd(home_folder); %change current folder to subject's folder
save(file_models,'RFmodel','HMMmodel')
disp('RF and HMM models saved.')
fprintf('\n')

%% RUN RANDOM FOREST ON FEATURES
tic
disp('Predicting with RF')
[codesRF,P_RF] = predict(RFmodel,Xall);
t = toc;
disp(['RF Prediction took ' num2str(t) ' seconds.'])
codesRF = str2num(cell2mat(codesRF));

if UseHMM
    disp('Predicting with HMM')
    tic
    [gamma, ~, ~, ~, ~]   = hmmInferNodes(HMMmodel,P_RF');
    [statesHmm, codesHmm] = getPredictedStates(gamma',uniqStates);
    
    timeHMM = toc;
    disp(['HMM Prediction took ' num2str(timeHMM) ' seconds.'])
    
    %do not rename the variables
    P_RF1 = P_RF;
    P_RF = gamma';      %the posteriors
    codesRF = codesHmm; %the predictions    
end

%% FIGURES SUMMARIZING RESULTS
home_folder = [ARCode 'home_data/CBR' subj_str '/'];
cd(home_folder)
addpath([code_folder '/export_figure/'])

%Pie Chart
maxPRF = max(P_RF,[],2);    %max posterior probability for each clip
ap = zeros(size(StateCodes,1),1);
posta = cell(size(ap));                     %RF post distr for each class
mposta = []; stdposta = [];
for a = 1:length(ap)
    inda = find(codesRF == a);
    ap(a) = length(inda)/length(codesRF)*100;  % perc. of time in each activity
    posta{a} = P_RF(inda,a);                   % prob distr of selected class a
    mposta = [mposta; mean(posta{a})];         % mean and std of prob distr
    stdposta = [stdposta; std(posta{a})];
    piepercent{a} =  {[num2str(ap(a),3) '%']};
    pielabel{a} =  StateCodes{a,1};
end
figure('name','Percent of Activities')
pchart = pie(ap,piepercent);
legend(pielabel,'Location','southoutside','Orientation','horizontal')
%set(pchart,'FontSize',16)
print(['CBR' subj_str '_' upper(brace_analyze) '_' file_end(7:end) '_Pie'],'-dpng')

%Confidence Plots (Box + Bar)
figure('name','Prob Distribution for Each Activity')
hold on
subplot(211)
boxplot(maxPRF,codesRF,'labels',StateCodes(:,1))
ylim([0 1.2])
subplot(212)
hold on
bar(1:length(StateCodes),mposta)
ax = gca; 
set(ax,'XTick',1:5,'XTickLabel',StateCodes(:,1))
ylim([0 1.2])
errorbar(1:length(StateCodes),mposta,stdposta,'ko')
print(['CBR' subj_str '_' upper(brace_analyze) '_' file_end(7:end) '_Prob'],'-dpng')

%Activity Profile (LINE)
if use_home
    Clipdur = cData_home.clipdur;                  %clip length in secs
else
    Clipdur = cData_lab.clipdur;                  %clip length in secs
end
Sf = 30;                                  %the sampling freq [Hz]
dur = Clipdur*Sf;    %the clip length [samples]
for ii = 1:length(time_clips) %crop all timestamps
    k2 = strfind(time_clips(ii,1),' ');
    k2 = cell2mat(k2); %convert string to number
    time_clips_cropped{ii} = time_clips{ii}(1:(k2-cut)); %crop to month/day
end
activity_tally = zeros(d,length(days));
days_crop = cellfun(@(s) s(1:end-1), days, 'uni',false);
time_match = 0;
for ii = 1:length(codesRF) %tally the activities
    time_match = find(ismember(days_crop,time_clips_cropped(ii)));
    activity_tally(codesRF(ii),time_match) = activity_tally(codesRF(ii),time_match) + 1;
end
days_plot = cellfun(@(s) s(1:end-5), days_crop, 'uni',false);
figure('name','Activity Profile by Calendar','units','normalized','outerposition',[0 0 1 1])
if data_removed == 0
    title({['CBR' subj_str ' - ' upper(brace_analyze)],['Data Removed: ' data_removed_str '%']},'FontSize',18)
else
    title({['CBR' subj_str ' - ' upper(brace_analyze)],['Data Removed: ' data_removed_str(1:5) '%']},'FontSize',18)
end
for ii = 1:length(uniqStates)
    hold on
    activity_tally_HR = activity_tally.*(6/3600); %convert # of clips to hours
    plot(activity_tally_HR(ii,:),'LineWidth',2)
    ylabel('Hours of Activity','FontSize',18)
    xlim([1 length(days_plot)])
    ylim([0 26])
    %ylim([0 (max(max(activity_tally_HR)).*1.2)])
    set(gca,'Box','off','XTick',[1:length(days_plot)],'XTickLabel',days_plot,'YTick',[4:4:24],'TickDir','out','LineWidth',2,'FontSize',14,'FontWeight','bold','XGrid','on');
    legend({StateCodes{:,1}},'FontSize',16)
end
export_fig(['CBR' subj_str '_' upper(brace_analyze) '_' file_end(7:end) '_Days.png'])
hold off

%Activity Profile (BAR)
figure('name','Activity Profile by Calendar (BAR)','units','normalized','outerposition',[0 0 1 1])
bar(activity_tally_HR','stacked')
if data_removed == 0
    title({['CBR' subj_str ' - ' upper(brace_analyze)],['Data Removed: ' data_removed_str '%']},'FontSize',18)
else
    title({['CBR' subj_str ' - ' upper(brace_analyze)],['Data Removed: ' data_removed_str(1:5) '%']},'FontSize',18)
end
ylabel('Hours of Activity','FontSize',18)
xlim([0 length(days_plot)+1])
ylim([0 26])
set(gca,'Box','off','XTick',[1:length(days_plot)],'XTickLabel',days_plot,'YTick',[4:4:24],'TickDir','out','LineWidth',2,'FontSize',14,'FontWeight','bold','XGrid','on');
legend({StateCodes{:,1}},'FontSize',16)
export_fig(['CBR' subj_str '_' upper(brace_analyze) '_' file_end(7:end) '_Days_Bar.png'])

%Stairs Per Day (BOX)
if length(uniqStates) == 5
    figure('name','Stairclimbing Time Distribution','units','normalized','outerposition',[0 0 1 1])
    Up = activity_tally_HR(2,:).*60; %convert to minutes
    Dw = activity_tally_HR(3,:).*60; %convert to minutes
    Stairs = [Up' Dw'];
    boxplot(Stairs,'labels',StateCodes(2:3,1))
    medstairs = median(sum(Stairs,2))
    title({['Distribution of Stairclimbing Time | CBR' subj_str ' - ' upper(brace_analyze)], ['Median = ' num2str(medstairs) ' min']},'FontSize',18)
    ylabel('Time [min]','FontSize',18)
    set(gca,'Box','off','TickDir','out','LineWidth',2,'FontSize',14,'FontWeight','bold');
    export_fig(['CBR' subj_str '_' upper(brace_analyze) '_' file_end(7:end) '_DistrStairs.png'])
end

%Activity Profile (BAR) - Normalized by Day to Percentages
activity_tally_HR_norm = zeros(size(activity_tally_HR));
summed = sum(activity_tally_HR,1); %totals for each day
for zz = 1:length(summed)
    if summed(zz) == 0
        %don't normalize (will be undefined values)
    else
        activity_tally_HR_norm(:,zz) = (activity_tally_HR(:,zz)./summed(zz))*100;
    end
end
figure('name','Activity Profile by Calendar (BAR_Norm)','units','normalized','outerposition',[0 0 1 1])
bar(activity_tally_HR_norm','stacked')
if data_removed == 0
    title({['CBR' subj_str ' - ' upper(brace_analyze)],['Data Removed: ' data_removed_str '%']},'FontSize',18)
else
    title({['CBR' subj_str ' - ' upper(brace_analyze)],['Data Removed: ' data_removed_str(1:5) '%']},'FontSize',18)
end
ylabel('Percent of Day','FontSize',18)
xlim([0 length(days_plot)+1])
ylim([0 100])
set(gca,'Box','off','XTick',[1:length(days_plot)],'XTickLabel',days_plot,'YTick',[10:10:100],'TickDir','out','LineWidth',2,'FontSize',14,'FontWeight','bold','XGrid','on');
legend({StateCodes{:,1}},'FontSize',16)
export_fig(['CBR' subj_str '_' upper(brace_analyze) '_' file_end(7:end) '_Days_Bar_Norm.png'])

%% RESET CURRENT DIRECTORY

cd(code_folder) %set current path back to code folder