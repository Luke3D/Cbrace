%% Thresholding for Unknown Data
% Aakash Gupta (November 21, 2015)
%INPUTS: Load features of unknown activities
%OUTPUTS: Optimized thresholds

%% LOAD DIRECTORIES AND INITIALIZE PARAMETERS
clear all, close all;

DSA_activity = [10:19]; %DSA activities to analyze
HAPT_subjects = []; %HAPT subjects to analyze
disp(['Analyzing ' num2str(length(DSA_activity)) ' DSA activities.'])
disp(['Analyzing ' num2str(length(HAPT_subjects)) ' HAPT subjects.'])
fprintf('\n')

p = gcp('nocreate');
if isempty(p)
    parpool('local')
end

cd(fileparts(which('unknownThreshold.m')))
currentDir = pwd;
slashdir = '/';
addpath([pwd slashdir 'sub']); %create path to helper scripts
cd ../
ARCode = pwd;
code_folder = [ARCode '/code/'];
home_data_folder = [ARCode '/home_data/'];
unk_folder = [ARCode '/unknown_data/'];
DSA_folder = [ARCode '/unknown_data/DSA/'];
DSA_raw_folder = [ARCode '/unknown_data/DSA/RawData/'];
DSA_feat_folder = [ARCode '/unknown_data/DSA/Features/'];
HAPT_folder = [ARCode '/unknown_data/HAPT/'];
HAPT_raw_folder = [ARCode '/unknown_data/HAPT/RawData/'];
HAPT_feat_folder = [ARCode '/unknown_data/HAPT/Features/'];
addpath(unk_folder)
addpath(DSA_folder)
addpath(DSA_raw_folder)
addpath(DSA_feat_folder)
addpath(HAPT_folder)
addpath(HAPT_raw_folder)
addpath(HAPT_feat_folder)
addpath(code_folder)
addpath(home_data_folder);

plotON = 1;                             %draw plots
drawplot.activities = 0;                % show % of each activity
drawplot.accuracy = 0;
drawplot.actvstime = 0;
drawplot.confmat = 0;

%Additional options
clipThresh = 0; %to be in training set, clips must have >X% of label
OOBVarImp = 'off';   %enable variable importance measurement

%% LOAD DATA TO ANALYZE
proceed = 1;
while proceed > 0
    population = input('Are you analyzing healthy or patient? ','s');
    if strcmpi(population,'patient')
        proceed = 0;
    elseif strcmpi(population,'healthy')
        proceed = 0;
    else
        disp('Please type healthy or patient.');
        proceed = 1;
    end
end

filename = ['trainData_' population '.mat'];
load(filename)

%% User Input for Which Subject to Analyze
tt = num2str(unique(trainingClassifierData.subjectID)');
fprintf('\n')
fprintf('Subject IDs present for analysis: %s',tt)
fprintf('\n')
fprintf('Available files to analyze: ')
fprintf('\n')
disp(unique(trainingClassifierData.subject))
fprintf('\n')

all_subjectID = trainingClassifierData.subjectID;

proceed = 1;
while proceed > 0 
    subject_analyze = input('Subject ID to analyze (ex. 5): ');

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

cData_temp2 = isolateSubject(trainingClassifierData,subject_indices);

if strcmpi(population,'patient')
    for zz = 1:length(cData_temp2.subject)
        temp = char(cData_temp2.subject(zz));
        cData_temp2.subjectBrace(zz) = {temp(7:9)};
    end

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

    cData_temp = isolateBrace(cData_temp2,brace_analyze);
else
    cData_temp = cData_temp2;
end

proceed = 1;
while proceed > 0
    fprintf('\n')
    disp('Please enter the max number of sessions to analyze.'); 
    disp('Or type 0 to analyze all sessions available.')
    min_sessions = input('Min session ID: ');
    max_sessions = input('Max session ID: ');
    proceed = 0;
end

if max_sessions == 0
    cData = cData_temp;
else
    cData = isolateSession(cData_temp,max_sessions,min_sessions);
end

fprintf('\n')
disp('These are the subjects that will be analyzed: ')
disp(unique(cData.subject))
fprintf('\n')

%% FILTER DATA FOR APPROPRIATE CLASSIFICATION RUNTYPE

%Create local variables for often used data
features     = cData.features; %features for classifier
subjects     = cData.subject;  %subject number
uniqSubjects = unique(subjects); %list of subjects
statesTrue = cData.activity;     %all the classifier data
uniqStates  = unique(statesTrue);          %set of states we have

%Remove any clips that don't meet the training set threshold
%This is the %80 threshold in the paper
[cData, removeInd] = removeDataWithActivityFraction(cData,clipThresh);

%How many clips of each activity type we removed
for i = 1:length(uniqStates)
    indr = find(strcmp(trainingClassifierData.activity(removeInd),uniqStates(i)));
    indtot = find(strcmp(trainingClassifierData.activity,uniqStates(i)));
    removed = length(indr)/length(indtot)*100;
    disp([num2str(removed) ' % of ' uniqStates{i} ' Train data removed (<' num2str(clipThresh) '% of clip)'])
end

%Get codes for the true states (i.e. make a number code for each state) and save code and state
codesTrue = zeros(1,length(statesTrue));
for i = 1:length(statesTrue)
    codesTrue(i) = find(strcmp(statesTrue{i},uniqStates));
end

%% CHECK FOR STAIRS IN EACH SESSION + REMOVE SESSOINS WITHOUT STAIRS

no_stairs_up = 0;
no_stairs_dw = 0;
sessions = unique(cData.sessionID);
sessions_removed = zeros(length(sessions),1);

for b = 1:length(sessions) %go through each session
    no_stairs_dw = isempty(find(cData.sessionID == sessions(b) & codesTrue' == 2, 1));
    no_stairs_up = isempty(find(cData.sessionID == sessions(b) & codesTrue' == 3, 1));
    
    if no_stairs_dw || no_stairs_up %remove session if no stairs data present in the session
        session_ind = find(cData.sessionID == sessions(b));
        features(session_ind,:) = [];
        subjects(session_ind) = [];
        statesTrue(session_ind) = [];
        codesTrue(session_ind) = [];
        cData.subjectID(session_ind) = [];
        cData.sessionID(session_ind) = [];
        cData.activity(session_ind) = [];
        cData.wearing(session_ind) = [];
        cData.identifier(session_ind) = [];
        cData.subject(session_ind) = [];
        cData.features(session_ind,:) = [];
        cData.activityFrac(session_ind) = [];
        cData.states(session_ind) = [];
        cData.subjectBrace(session_ind) = [];
        
        sessions_removed(b) = 1;
    end
end

fprintf('\n')
if any(sessions_removed == 1)
    disp('Not all sessions have stairs...')
    disp(['Session IDs Removed: '])
    disp(sessions(logical(sessions_removed)))
else
    disp('All sessions have stairs...')
    disp(['Session IDs Removed: NONE'])
end

%% SORT THE DATA FOR K-FOLDS + RF TRAIN/TEST

%Indices for test set + set all to 0, we will specify test set soon
testSet = false(length(statesTrue),1);

%Store Code and label of each unique State
StateCodes = cell(length(uniqStates),2);
StateCodes(:,1) = uniqStates;
StateCodes(:,2) = num2cell(1:length(uniqStates)); %sorted by unique

%SessionID index ranges (differentiate CBR session 1 from SCO session 1)
N_session = length(cData.sessionID);
ind_change = [1]; %include first index;
for kk = 1:(N_session-1)
    if ~((cData.sessionID(kk+1))-(cData.sessionID(kk)) == 0)
        ind_change = [ind_change kk]; %store all the sessionID change indices
    end
end
ind_change = [ind_change N_session]; %include last index

folds = length(ind_change) - 1; %number of folds = number of sessions
activity_acc_matrix = zeros(5,folds);

%% Import Unknown Data + Split into k-folds
X_unk = [];

%Import DSA Features
for kk = 1:length(DSA_activity)
    if DSA_activity(kk) < 10
        unk_subj_str = ['0' num2str(DSA_activity(kk))];
    elseif DSA_activity(kk) > 9
        unk_subj_str = num2str(DSA_activity(kk));
    end
    
    file_unk = [DSA_feat_folder 'DSA_' unk_subj_str '.mat'];
    load(file_unk)
    X_unk = [X_unk; X_all];
end

%Import HAPT Features
for kk = 1:length(HAPT_subjects)
    if HAPT_subjects(kk) < 10
        unk_subj_str = ['0' num2str(HAPT_subjects(kk))];
    elseif HAPT_subjects(kk) > 9
        unk_subj_str = num2str(HAPT_subjects(kk));
    end
    
    file_unk = [HAPT_feat_folder 'HAPT_' unk_subj_str '.mat'];
    load(file_unk)
    X_unk = [X_unk; X_all];
end

%Randomly shuffle total features matrix
X_unk = X_unk(randperm(length(X_unk)),:);

%Split data into folds
ind_change_2 = [1:floor(length(X_unk)/folds):length(X_unk)];
ind_change_2(end) = length(X_unk);
if length(ind_change)~=length(ind_change_2)
    ind_change_2 = [1:floor(length(X_unk)/folds):length(X_unk)];
    ind_change_2(end+1) = length(X_unk);
end
    
%% RUN RANDOM FOREST ON k-FOLDS

%Do k fold cross validation for RF
for k = 1:folds
    %% Create Train and Test vector - Split dataset into k-folds
    testSet = zeros(length(statesTrue),1);
    testSet(ind_change(k):ind_change(k+1)) = 1;
    testSet = logical(testSet);
    
    testSet_unk = zeros(length(X_unk),1);
    testSet_unk(ind_change_2(k):ind_change_2(k+1)) = 1;
    testSet_unk = logical(testSet_unk);
    
    %Remove clips that are a mix of activities from training set
    %These were the clips that did not meet the 80% threshold
    TrainingSet = ~testSet;
    TrainingSet(removeInd) = 0;   %remove clips
    
    %% RF TRAINING AND TESTING
    %TRAIN RF
    ntrees = 300;
    
    %cost matrix 
    CostM = 0.5*ones(5,5); CostM([1 7 13 19 25]) = 0; 
    CostM(2,:) = 5; CostM(2,2) = 0;    %increase cost of misclassifying stairs
    CostM(3,:) = 5; CostM(3,3) = 0;
    
    tic; %start timer
    
    fprintf('\n')
    disp(['RF Train - Fold '  num2str(k) '  #Samples Train = ' num2str(N_session - (ind_change(k+1)-ind_change(k))) '  #Samples Test = ' num2str((ind_change(k+1)-ind_change(k)))]);
    disp('Training RF model.');
    
    %How many samples of each activity we have in the test fold
    for i = 1:length(uniqStates)
        inda = find(strcmp(cData.activity(testSet),uniqStates(i)));
        Nsa = length(inda);
        indtot = find(strcmp(trainingClassifierData.activity,uniqStates(i)));
        Nsaperc = Nsa/length(indtot)*100;
        disp([num2str(Nsa) ' Samples of ' uniqStates{i} ' in this fold  (' num2str(Nsaperc) '% of total)'])
    end

    opts_ag = statset('UseParallel',1);
    RFmodel = TreeBagger(ntrees,features(TrainingSet,:),codesTrue(TrainingSet)','OOBVarImp',OOBVarImp,'Cost',CostM,'Options',opts_ag); 
    
    timeRF = toc; %end timer
    thresh = zeros(length(uniqStates),1);
      
    %% Predict on Known Data
    disp('Predicting on known data.')
    [codesRF_kn,P_RF_kn] = predict(RFmodel,features(testSet,:));    
    [M_kn, I_kn] = max(P_RF_kn,[],2);
    
    %% Predict on Unknown Data
    disp('Predicting on unknown data.')
    [codesRF_unk,P_RF_unk] = predict(RFmodel,X_unk(testSet_unk,:));
    [M_unk, I_unk] = max(P_RF_unk,[],2);
    
    %% Optimize Threshold Values
    disp('Optimizing thresholds.')
    it = [(1/length(uniqStates)):0.01:1]; %Threshold iterations from lowest posterior to maximum of one
        
    %Known Data
    n_total_kn = length(M_kn); %length of known data
    kn_mat = zeros(length(it),length(thresh)); %record accuracies for each threshold iteration
    for ii = 1:length(thresh)
        thresh_it = zeros(5,1);
        for tt = 1:length(it)
            thresh_it(ii) = it(tt);
            kn_mat(tt,ii) = (length(find(I_kn == ii & M_kn > thresh_it(I_kn))))./length(find(I_kn == ii)); %threshold posteriors for the current iteration
        end
    end

    %Unknown Data
    n_total_unk = length(M_unk); %length of unknown data
    unk_mat = zeros(length(it),length(thresh)); %record accuracies for each threshold iteration
    for ii = 1:length(thresh)
        thresh_it = zeros(5,1);
        for tt = 1:length(it)
            thresh_it(ii) = it(tt);
            unk_mat(tt,ii) = (length(find(I_unk == ii & M_unk < thresh_it(I_unk))))./length(find(I_unk == ii)); %threshold posteriors for the current iteration
        end
    end
    
    %% Plot Optimization Graph
    figure;
    it_max = zeros(size(unk_mat,2),1);
    for b = 1:size(unk_mat,2)
        subplot(1,5,b)
        hold on
        plot(it, kn_mat(:,b),'LineWidth',3);
        plot(it, unk_mat(:,b),'LineWidth',3);

        %Plot maximum for both
        total = kn_mat(:,b) + unk_mat(:,b); %sum both known and unknown accuracies
        [~,it_ind] = max(total); %OPTIMIZE accuracy based on maximum of sum
        yL_2 = [0 1];
        line([it(it_ind) it(it_ind)],yL_2,'Color','r');
        it_max(b) = it(it_ind);

        legend('Known','Unknown')
        xlabel('Threshold Value','FontSize',18)
        ylabel('Accuracy','FontSize',18)
        title({StateCodes{b,1},['Known: ' num2str(100*length(find(I_kn == b))./n_total_kn) '%'],['Unknown: ' num2str(100*length(find(I_unk == b))./n_total_unk) '%']},'FontSize',20)
        set(gca,'Box','off','TickDir','out','LineWidth',2,'FontSize',14,'FontWeight','bold','XGrid','on');
        xlim([0 1])
        ylim([0 1])
        hold off
    end

    %% Plot Confusion Matrix for Optimized Threshold
    mat_kn_unk = zeros(2,2);
    mat_kn_unk(1,1) = length(find(M_kn > it_max(I_kn)))./n_total_kn;
    mat_kn_unk(2,2) = length(find(M_unk < it_max(I_unk)))./n_total_unk;
    mat_kn_unk(1,2) = 1 - mat_kn_unk(1,1);
    mat_kn_unk(2,1) = 1 - mat_kn_unk(2,2);
    
    acc_kn = mat_kn_unk(1,1);
    acc_unk = mat_kn_unk(2,2);
    
    figure('name',['k-fold ' num2str(k)]);
    imagesc(mat_kn_unk);
    colorbar
    [cmin,cmax] = caxis;
    caxis([0,1])
    ax = gca;
    ax.XTick = 1:2;
    ax.YTick = 1:2;
    xlabel('Predicted','FontSize',18)
    ylabel('True','FontSize',18)
    set(gca,'XTickLabel',{'Known','Unknown'},'YTickLabel',{'Known','Unknown'},'TickDir','out','LineWidth',2,'FontSize',14,'FontWeight','bold')
    axis square
     
    %% Store Results
    results(k).acc_kn = mat_kn_unk(1,1);
    results(k).acc_unk = mat_kn_unk(2,2);    
    results(k).mat_kn_unk = mat_kn_unk;
    results(k).M_kn = M_kn;
    results(k).M_unk = M_unk;
    results(k).I_kn = I_kn;
    results(k).I_unk = I_unk;
    results(k).it_max = it_max;
     
    disp(['Known Accuracy = ' num2str(acc_kn)]);   
    disp(['Unknown Accuracy = ' num2str(acc_unk)]);
end

%% Calculate and Plot Final Results
%Calculate Average Accuracy Across Folds
acc_kn = 0; acc_unk = 0;
for i = 1:folds
    acc_kn = acc_kn + results(i).acc_kn.*(ind_change(i+1)-ind_change(i)); %weighted average
    acc_unk = acc_unk + results(i).acc_unk.*(ind_change_2(i+1)-ind_change_2(i)); %weighted average
end
acc_kn = acc_kn/N_session;
acc_unk = acc_unk/length(X_unk);
fprintf('\n')
disp(['Weighted Mean Known Accuracy = ' num2str(acc_kn)]);
disp(['Weighted Mean Unknown Accuracy = ' num2str(acc_unk)]);

%Mean Confusion Matrix
mat_avg = zeros(2,2);
mat_avg(1,1) = acc_kn;
mat_avg(2,2) = acc_unk;
mat_avg(1,2) = 1 - mat_avg(1,1);
mat_avg(2,1) = 1 - mat_avg(2,2);
figure('name','Mean Confusion Matrix');
imagesc(mat_avg);
colorbar
[cmin,cmax] = caxis;
caxis([0,1])
ax = gca;
ax.XTick = 1:2;
ax.YTick = 1:2;
xlabel('Predicted','FontSize',18)
ylabel('True','FontSize',18)
set(gca,'XTickLabel',{'Known','Unknown'},'YTickLabel',{'Known','Unknown'},'TickDir','out','LineWidth',2,'FontSize',14,'FontWeight','bold')
axis square

%Average Thresholds
it_avg = zeros(length(uniqStates),1);
for r = 1:folds
    it_avg = it_avg + results(r).it_max;
end
it_avg = it_avg./folds; %average optimized thresholds across folds

%Posterior Distributions with Optimized Threshold Plotted
post_mat = cell(2,length(uniqStates));
for c = 1:folds
    for b = 1:length(uniqStates)
        t1 = find(results(c).I_kn==b);
        t2 = find(results(c).I_unk==b);
        post_mat{1,b} = [post_mat{1,b}; results(c).M_kn(t1)];
        post_mat{2,b} = [post_mat{2,b}; results(c).M_unk(t2)];
    end
end
figure('name','Posterior Distribution');
bins = 20; count = 1;
for m = 1:length(uniqStates)
    [k1, ~] = histcounts(post_mat{1,m},bins);
    [k2, ~] = histcounts(post_mat{2,m},bins);
    y_max = max([k1 k2])*1.1;
    
    subplot(2,length(uniqStates),count)
    histogram(post_mat{1,m},bins)
    title({['Known - ' uniqStates{m}],['N = ' num2str(length(post_mat{1,m}))]})
    xlim([0 1])
    ylim([0 y_max])
    yL = get(gca,'YLim');
    line([it_avg(m) it_avg(m)],yL,'Color','r');
    
    subplot(2,length(uniqStates),count+length(uniqStates))
    histogram(post_mat{2,m},bins)
    title({['Unknown - ' uniqStates{m}],['N = ' num2str(length(post_mat{2,m}))]})
    xlim([0 1])
    ylim([0 y_max])
    yL = get(gca,'YLim');
    line([it_avg(m) it_avg(m)],yL,'Color','r');
    
    count = count + 1;
end

fprintf('\n')
disp('OPTIMIZED THRESHOLDS:')
it_avg

%% Export Averaged Optimized Thresholds
if subject_analyze < 10
    subj_str = ['0' num2str(subject_analyze)];
elseif subject_analyze > 9
    subj_str = num2str(subject_analyze);
end
filename = [home_data_folder 'CBR' subj_str '/' upper(brace_analyze) '_THRESH.mat'];
save(filename,'it_avg','acc_kn','acc_unk') %saves final thresholds for speicific person+brace