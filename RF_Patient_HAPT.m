%% RF for Training on HAPT and Testing on Patients
%Aakash Gupta

%% LOAD DATA AND INITIALIZE PARAMETERS
clear all, close all;

p = gcp('nocreate');
if isempty(p)
    parpool('local')
end

cd(fileparts(which('RF_Patient_HAPT.m')))
currentDir = pwd;
slashdir = '/';
addpath([pwd slashdir 'sub']); %create path to helper scripts
addpath(genpath([slashdir 'Traindata'])); %add path for train data

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

plotON = 1;                             %draw plots
drawplot.activities = 1;                % show % of each activity
drawplot.accuracy = 0;
drawplot.actvstime = 0;
drawplot.confmat = 1;

%Additional options
clipThresh = 0; %to be in training set, clips must have >X% of label
OOBVarImp = 'off';   %enable variable importance measurement

%% LOAD HAPT DATA + TRAIN CLASSIFIER
HAPT_subjects = [1:30];

%Import Activity Labels
% filename = [HAPT_raw_folder 'labels.txt'];
% delimiter = ' ';
% formatSpec = '%f%f%f%f%f%[^\n\r]';
% fileID = fopen(filename,'r');
% dataArray = textscan(fileID, formatSpec, 'Delimiter', delimiter, 'MultipleDelimsAsOne', true,  'ReturnOnError', false);
% fclose(fileID);
% unk_labels = [dataArray{1:end-1}];
% clear dataArray

%Import Features
X_unk = [];
unk_labels = [];
for kk = 1:length(HAPT_subjects)
    if HAPT_subjects(kk) < 10
        unk_subj_str = ['0' num2str(HAPT_subjects(kk))];
    elseif HAPT_subjects(kk) > 9
        unk_subj_str = num2str(HAPT_subjects(kk));
    end
    
    file_unk = [HAPT_feat_folder 'HAPT_' unk_subj_str '.mat'];
    load(file_unk)
    X_unk = [X_unk; X_all];
    unk_labels = [unk_labels; X_labels];
end

features_h = X_unk;
activity_h = unk_labels;
codesTrue_h = activity_h;

%cost matrix
CostM = 0.5*ones(5,5); CostM([1 7 13 19 25]) = 0;
CostM(2,:) = 5; CostM(2,2) = 0;    %increase cost of misclassifying stairs
CostM(3,:) = 5; CostM(3,3) = 0;

%Train RF
disp(['Training classifier on ' num2str(length(HAPT_subjects)) ' healthy HAPT subjects...'])
ntrees = 300;
opts_ag = statset('UseParallel',1);
RFmodel = TreeBagger(ntrees,features_h,codesTrue_h','OOBVarImp',OOBVarImp,'Cost',CostM,'Options',opts_ag);
disp('Classifier trained.')

%% LOAD PATIENT DATA TO ANALYZE
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
disp('Please enter subject IDs to analyze one at a time.')
disp('When you are done type 0.')
fprintf('\n')
pause(1)

user_subjects = [];
subject_indices = [];
proceed = 1;
while proceed > 0 
    subject_analyze = input('Subject ID to analyze (ex. 5): ');

    if (subject_analyze == 0)
        proceed = 0;
    else
        %Check if subjectID is in mat file
        if ~any(subject_analyze == all_subjectID)
            disp('-------------------------------------------------------------')
            disp('Subject ID not in trainingClassifierData.mat file. Try again.')
            disp('-------------------------------------------------------------')
        else
            subject_indices = [subject_indices; find(subject_analyze==all_subjectID)];
            user_subjects = [user_subjects subject_analyze];
        end
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

%Remove data from other locations if required (old datasets)
cData = removeDataWithoutLocation(cData,'Belt');

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

%% SORT THE DATA FOR K-FOLDS + RF TRAIN/TEST

%Indices for test set + set all to 0, we will specify test set soon
testSet = false(length(statesTrue),1);

%Get codes for the true states (i.e. make a number code for each state) and save code and state
codesTrue = zeros(1,length(statesTrue));
for i = 1:length(statesTrue)
    codesTrue(i) = find(strcmp(statesTrue{i},uniqStates));
end

%Store Code and label of each unique State
StateCodes = cell(length(uniqStates),2);
StateCodes(:,1) = uniqStates;
StateCodes(:,2) = num2cell(1:length(uniqStates)); %sorted by unique

%SubjectID index ranges
N_session = length(cData.subjectID);
ind_change = [1]; %include first index;
for kk = 1:(N_session-1)
    if ~((cData.subjectID(kk+1))-(cData.subjectID(kk)) == 0)
        ind_change = [ind_change kk]; %store all the sessionID change indices
    end
end
ind_change = [ind_change N_session]; %include last index

folds = length(user_subjects); %number of folds = number of sessions
activity_acc_matrix = zeros(5,folds);

%Do k fold cross validation for RF
for k = 1:folds
    %% Create Train and Test vector - Split dataset into k-folds
    testSet = zeros(length(statesTrue),1);
    testSet(ind_change(k):ind_change(k+1)) = 1;
    testSet = logical(testSet);
    
    %Remove clips that are a mix of activities from training set
    %These were the clips that did not meet the 80% threshold
    TrainingSet = ~testSet;
    TrainingSet(removeInd) = 0;   %remove clips
    
    %% RF TRAINING AND TESTING
       
    fprintf('\n')
    disp(['RF Train - Fold '  num2str(k) '  #Samples Train = ' num2str(length(activity_h)) '  #Samples Test = ' num2str((ind_change(k+1)-ind_change(k)))]);
    
    %How many samples of each activity we have in the test fold
    for i = 1:length(uniqStates)
        inda = find(strcmp(cData.activity(testSet),uniqStates(i)));
        Nsa = length(inda);
        indtot = find(strcmp(trainingClassifierData.activity,uniqStates(i)));
        Nsaperc = Nsa/length(indtot)*100;
        disp([num2str(Nsa) ' Samples of ' uniqStates{i} ' in this fold  (' num2str(Nsaperc) '% of total)'])
    end
      
    %RF Prediction and RF class probabilities for ENTIRE dataset. This is
    %for initializing the HMM Emission matrix (P_RF(TrainSet)) and for
    %computing the observations of the HMM (P_RF(TestSet))
    [codesRF,P_RF] = predict(RFmodel,features_h);
    codesRF = str2num(cell2mat(codesRF));
    statesRF = uniqStates(codesRF);
    
    %% RESULTS for each k-fold
    %entire classification matrix (HMM prediction is run only on Test data)
    [matRF,accRF,labels] = createConfusionMatrix(codesTrue(testSet),codesRF(testSet));
    
    %Store all results
    results(k).stateCodes        = StateCodes;
    %OVERALL ACCURACY
    results(k).accRF             = accRF;
    %CONFUSION MATRICES
    results(k).matRF             = matRF;
    %PRED and ACTUAL CODES AND STATES
    results(k).statesRF         = statesRF(testSet);
    results(k).statesTrue       = statesTrue(testSet);
    results(k).trainingSet      = TrainingSet;
    results(k).testSet          = testSet;
%     results(k).codesTrue       = codesTrue(testSet);
%     results(k).codesRF         = codesRF(testSet);
%     results(k).codesHmm        = codesHmm;
     
    disp(['accRF = ' num2str(accRF)]);
    
    %% PLOT PREDICTED AND ACTUAL STATES
    
    %Rename states for plot efficiency
    StateCodes(:,1) = {'Sit','Stairs Dw','Stairs Up','Stand','Walk'};
    nstates = length(StateCodes);

    if plotON
        if drawplot.actvstime
            dt = cData.clipdur * (1-cData.clipoverlap);
            t = 0:dt:dt*(length(codesTrue(testSet))-1);  
            figure('name',['k-fold ' num2str(k)]); hold on
            subplot(211), hold on
            plot(t,codesTrue(testSet),'.-g')
            plot(t,codesRF(testSet)+.1,'.-r')
            xlabel('Time [s]')
            legend('True','RF')
            ylim([0.5 nstates+0.5]);
            set(gca,'YTick',cell2mat(StateCodes(:,2))')
            set(gca,'YTickLabel',StateCodes(:,1))
            subplot(212)
            plot(t,max(P_RF(testSet,:),[],2),'r'), hold on
            line([0 t(end)],[1/nstates 1/nstates])
        end
            
        if drawplot.confmat
            figure('name',['k-fold ' num2str(k)]); hold on
            correctones = sum(matRF,2);
            correctones = repmat(correctones,[1 size(StateCodes,1)]);
            imagesc(matRF./correctones); colorbar
            set(gca,'XTick',[1:5],'XTickLabel',StateCodes(:,1))
            set(gca,'YTick',[1:5],'YTickLabel',StateCodes(:,1))
            axis square
            
            activity_acc_matrix(:,k) = diag(matRF./correctones);
        end
    end
end

%Display % of each activity over all predictions
if drawplot.activities
    Activity = uniqStates;
    for a = 1:size(StateCodes,1)
        ind = [];  %temp var to store activity found
        for k = 1:folds
            ind = [ind; strcmp(results(k).statesRF,Activity{a})];
        end
        Activity{a,2} = sum(ind)./size(ind,1)*100;  % the % of each activity 
    end 
    figure('name','% of activities')
    pchart = pie(cell2mat(Activity(:,2)),StateCodes(:,1));
%     bar(cell2mat(Activity(:,2)));
%     set(gca,'XTickLabel',StateCodes(:,1))
%     ylabel('% of time spent')
end

%Average accuracy over all folds (weighted and unweighted)
accRF = 0; matRF = zeros(nstates); accRF_uw = 0;
for i = 1:folds
    accRF = accRF + results(i).accRF.*(ind_change(i+1)-ind_change(i)); %weighted average
    correctones = sum(results(i).matRF,2);
    correctones = repmat(correctones,[1 size(StateCodes,1)]);
    matRF = matRF + (results(i).matRF)./correctones;
end
accRF_uw = sum([results(:).accRF])/folds;
accRF = accRF/N_session;
fprintf('\n')
disp(['Unweighted Mean (k-fold) accRF = ' num2str(accRF_uw)]);
disp(['Weighted Mean (k-fold) accRF = ' num2str(accRF)]);

matRF = matRF./folds;
figure('name','Mean (k-fold) Confusion matrix')
imagesc(matRF); 
colorbar
[cmin,cmax] = caxis;
caxis([0,1])
ax = gca;
ax.XTick = 1:size(StateCodes,1);
ax.YTick = 1:size(StateCodes,1);
set(gca,'XTickLabel',{})
set(gca,'YTickLabel',StateCodes(:,1))
axis square

%% Accuracy Table
test_size = zeros(folds+2,1);
train_size = zeros(folds+2,1);
row_folds = cell(folds+2,1);
for ii = 1:folds
    test_size(ii) = (ind_change(ii+1)-ind_change(ii));
    train_size(ii) = N_session - (ind_change(ii+1)-ind_change(ii));
    row_folds(ii) = {['Fold ' num2str(ii)]};
end

row_folds(end-1) = {'Mean (UW)'};
row_folds(end) =  {'Mean (W)'};

acc_RF_vector = [results.accRF accRF_uw accRF]';

acc_tbl = table(test_size,train_size,acc_RF_vector,'RowNames',row_folds,'VariableNames',{'Test_Size','Train_Size','RF_Acc'});
fprintf('\n')
disp(acc_tbl)

%% Activity Accuracy Table
temp2 = mean(activity_acc_matrix');
activity_acc = [activity_acc_matrix temp2'];
act = {'Sitting';'Stairs Dw';'Stairs Up';'Standing';'Walking'};
var_name = cell(folds+1,1);
for ii = 1:folds
    var_name(ii) = {['Fold_' num2str(ii)]};
end
var_name(end) = {'Mean'};

activity_tbl = array2table(activity_acc,'RowNames',act,'VariableNames',var_name);
disp(activity_tbl)