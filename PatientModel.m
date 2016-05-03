%% DIRECTORIES
clear all, close all;

patient_stairs = [2 8 11 12 15];
disp(patient_stairs);

p = gcp('nocreate');
if isempty(p)
    parpool('local')
end

cd(fileparts(which('PatientModel.m')))
currentDir = pwd;
slashdir = '/';
addpath([pwd slashdir 'sub']); %create path to helper scripts
addpath(genpath([slashdir 'Traindata'])); %add path for train data

plotON = 1;                             %draw plots
drawplot.activities = 0;                % show % of each activity
drawplot.accuracy = 0;
drawplot.actvstime = 0;
drawplot.confmat = 0;

%Additional options
clipThresh = 0; %to be in training set, clips must have >X% of label
OOBVarImp = 'off';   %enable variable importance measurement

%% LOAD DATA TO ANALYZE
population = 'patient';
filename = ['trainData_' population '.mat'];
load(filename)

%% SELECT PATIENT
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

%% EXTRACT MAIN SESSIONS (personal data)
states = {'Sitting';'Stairs Dw';'Stairs Up';'Standing';'Walking'};

features_main     = cData.features; %features for classifier
subjects_main     = cData.subject;  %subject number
uniqSubjects_main = unique(subjects_main); %list of subjects
statesTrue_main = cData.activity;     %all the classifier data
subjectID_main = cData.subjectID;
sessionID_main = cData.sessionID;

%Remove stairs data from specific patients
stairs_remove = [];
for h = 1:length(patient_stairs)
    a1 = find(subjectID_main == patient_stairs(h));
    a2 = strmatch('Stairs Up',statesTrue_main,'exact');
    a = intersect(a1,a2);
    
    b1 = find(subjectID_main == patient_stairs(h));
    b2 = strmatch('Stairs Dw',statesTrue_main,'exact');
    b = intersect(b1,b2);
    
    stairs_remove = [stairs_remove; a; b];
end
features_main(stairs_remove,:) = [];
subjects_main(stairs_remove) = [];
statesTrue_main(stairs_remove) = [];
subjectID_main(stairs_remove) = [];
sessionID_main(stairs_remove) = [];
uniqStates_main  = unique(statesTrue_main); 

%Generate codesTrue
codesTrue_main = zeros(1,length(statesTrue_main));
for i = 1:length(statesTrue_main)
    codesTrue_main(i) = find(strcmp(statesTrue_main{i},states));
end
disp('Data extracted for sessions of interest.')

%% EXTRACT NEW SESSION (personal data)
%Isolate data
sessionID_train = 4;
subject = isolateSession(cData_temp,sessionID_train,sessionID_train);

%Extract data
features_new     = subject.features; %features for classifier
subjects_new     = subject.subject;  %subject number
uniqSubjects_new = unique(subjects_new); %list of subjects
statesTrue_new = subject.activity;     %all the classifier data
subjectID_new = subject.subjectID;
sessionID_new = subject.sessionID;

%Remove stairs data from specific patients
stairs_remove = [];
for h = 1:length(patient_stairs)
    a1 = find(subjectID_new == patient_stairs(h));
    a2 = strmatch('Stairs Up',statesTrue_new,'exact');
    a = intersect(a1,a2);
    
    b1 = find(subjectID_new == patient_stairs(h));
    b2 = strmatch('Stairs Dw',statesTrue_new,'exact');
    b = intersect(b1,b2);
    u
    stairs_remove = [stairs_remove; a; b];
end
features_new(stairs_remove,:) = [];
subjects_new(stairs_remove) = [];
statesTrue_new(stairs_remove) = [];
subjectID_new(stairs_remove) = [];
sessionID_new(stairs_remove) = [];
uniqStates_new  = unique(statesTrue_new);

%Generate codesTrue
codesTrue_new = zeros(1,length(statesTrue_new));
for i = 1:length(statesTrue_new)
    codesTrue_new(i)  = find(strcmp(statesTrue_new{i},states));
end
disp('Data extracted for new session.')
fprintf('\n')

%% LAYER 1: TRAIN CLASSIFIERS + GENERATE POSTERIORS
%Isolate brace and sessions
for zz = 1:length(trainingClassifierData.subject)
    temp = char(trainingClassifierData.subject(zz));
    trainingClassifierData.subjectBrace(zz) = {temp(7:9)};
end
patients_temp_2 = isolateBrace(trainingClassifierData,brace_analyze);
patients_temp = isolateSession(patients_temp_2,max_sessions,min_sessions);

IDs = unique(trainingClassifierData.subjectID);
ind_temp = find(subject_analyze == IDs); %find patient to analyze
IDs(ind_temp) = []; %remove patient so list contains global patients
posteriors_new = []; %stores all posteriors to be generated
posteriors_main = []; %stores all posteriors to be generated
n_act = zeros(length(IDs),1);
n_train = zeros(length(IDs),1);
acc_new = zeros(length(IDs),1);
acc_main = zeros(length(IDs),1);
acc_act_new = cell(length(IDs),1); %stores acccuracies for each activity for each patient

disp('Initiating Layer 1...')
disp('Cycling through each global patient:')
%Cycle through each patient
for y = 1:length(IDs)
    
    %Isolate  subject
    subject_indices = find(IDs(y)==patients_temp.subjectID);
    patients = isolateSubject(patients_temp,subject_indices);
    
    %Extract data
    features_p     = patients.features; %features for classifier
    subjects_p     = patients.subject;  %subject number
    uniqSubjects_p = unique(subjects_p); %list of subjects
    statesTrue_p = patients.activity;     %all the classifier data
    subjectID_p = patients.subjectID;
    sessionID_p = patients.sessionID;
    
    %Remove stairs data from specific patients
    stairs_remove = [];
    for h = 1:length(patient_stairs)
        a1 = find(subjectID_p == patient_stairs(h));
        a2 = strmatch('Stairs Up',statesTrue_p,'exact');
        a = intersect(a1,a2);
        
        b1 = find(subjectID_p == patient_stairs(h));
        b2 = strmatch('Stairs Dw',statesTrue_p,'exact');
        b = intersect(b1,b2);
        
        stairs_remove = [stairs_remove; a; b];
    end
    features_p(stairs_remove,:) = [];
    subjects_p(stairs_remove) = [];
    statesTrue_p(stairs_remove) = [];
    subjectID_p(stairs_remove) = [];
    sessionID_p(stairs_remove) = [];
    uniqStates_p  = unique(statesTrue_p);
    
    n_act(y) = length(uniqStates_p);
    n_train(y) = size(features_p,1);
    
    %Generate codesTrue
    codesTrue_p = zeros(1,length(statesTrue_p));
    for i = 1:length(statesTrue_p)
        codesTrue_p(i)  = find(strcmp(statesTrue_p{i},states));
    end
    
    %Train Random Forest on Global Patient
    ntrees = 100;
    disp(['RF Train - Patient '  num2str(IDs(y)) '  #Samples Train = ' num2str(size(features_p,1))]);
    opts_ag = statset('UseParallel',1);
    RFmodel_p = TreeBagger(ntrees,features_p,codesTrue_p,'OOBVarImp',OOBVarImp,'Options',opts_ag);
    
    %Test Random Forest on Main Sessions
    [codesRF_main,P_RF_main] = predict(RFmodel_p,features_main);
    codesRF_main = str2num(cell2mat(codesRF_main));
    
    %Test Random Forest on New Session
    [codesRF_new,P_RF_new] = predict(RFmodel_p,features_new);
    codesRF_new = str2num(cell2mat(codesRF_new));
    
    %Collect Accuracies
    [matRF_main,acc_main(y),~] = createConfusionMatrix(codesTrue_main,codesRF_main);
    [matRF_new,acc_new(y),~] = createConfusionMatrix(codesTrue_new,codesRF_new);
    
    correctones = sum(matRF_main,2);
    correctones = repmat(correctones,[1 length(states)]);
    diagonal_main = diag(matRF_main./correctones);
    correctones = sum(matRF_new,2);
    correctones = repmat(correctones,[1 length(states)]);
    diagonal_new = diag(matRF_new./correctones);
    
%     if length(uniqStates_p) == 3
%         diagonal_main(2:3) = [];
%         diagonal_new(2:3) = [];
%     end
%     
%     for h = 1:size(P_RF_main,1)
%         P_RF_main(h,:) = (P_RF_main(h,:).*diagonal_main').*acc_main(y);
%     end
%     for h = 1:size(P_RF_new,1)
%         P_RF_new(h,:) = (P_RF_new(h,:).*diagonal_new').*acc_new(y);
%     end
    
    %Collect Posteriors
    posteriors_main = [posteriors_main P_RF_main];
    posteriors_new = [posteriors_new P_RF_new];
end
disp('Posteriors generated for main and new sessions.')

%% LAYER 1: GENERATE FEATURES
features_main = getFeaturesTR(posteriors_main);
features_new = getFeaturesTR(posteriors_new);
disp('Feature generation complete')
disp('Layer 1 complete.')
fprintf('\n')

%% LAYER 1: SUMMARY TABLE
layer1_tbl = table(IDs,n_act,n_train,acc_new,acc_main,'VariableNames',{'ID','N_Activities','Train_Size','Acc_New','Acc_Main'});
disp(layer1_tbl);

figure;
subplot(2,1,1)
hold on
plot(1:length(IDs),acc_new,'LineWidth',2)
plot(1:length(IDs),acc_main,'LineWidth',2)
ylim([0 1])
ylabel('Accuracy','FontSize',16)
set(gca,'Box','off','XTick',[1:length(IDs)],'XTickLabel',{},'YTick',[0.1:0.1:1],'TickDir','out','LineWidth',2,'FontSize',14,'FontWeight','bold');
legend({'Accuracy New','Accuracy Main'},'FontSize',16)
hold off

subplot(2,1,2)
plot(1:length(IDs),n_train,'LineWidth',2)
xlabel('C-Brace Subject ID','FontSize',16)
ylabel('Training Data Size','FontSize',16)
set(gca,'Box','off','XTick',[1:length(IDs)],'XTickLabel',num2cell(IDs),'TickDir','out','LineWidth',2,'FontSize',14,'FontWeight','bold');

%% LAYER 2: TRAIN CLASSIFIER + GENERATE FINAL ACTIVITY PREDICTIONS
%Train RF
ntrees = 100;
RFmodel = TreeBagger(ntrees,features_new,codesTrue_new,'OOBVarImp',OOBVarImp,'Options',opts_ag);

%Test RF
[codesRF_FINAL,P_RF_FINAL] = predict(RFmodel,features_main);
[matRF_FINAL,acc_RF_FINAL,~] = createConfusionMatrix(codesTrue_main,codesRF_FINAL);
disp(matRF_FINAL)
disp(acc_RF_FINAL)


% B = mnrfit(features_new,codesTrue_new);
% pihat = mnrval(B,features_main);
% [~, codesLR_FINAL] = max(pihat,[],2);
% [matRF_FINAL,acc_FINAL,~] = createConfusionMatrix(codesTrue_main,codesLR_FINAL);
% disp(acc_FINAL)

% %SVM
% options = statset('UseParallel',1);
% template = templateSVM('KernelFunction', 'linear', 'PolynomialOrder', [], 'KernelScale', 'auto', 'BoxConstraint', 1, 'Standardize', 1);
% trainedClassifier = fitcecoc(features_new, codesTrue_new, 'Learners', template, 'FitPosterior', 1, 'Coding', 'onevsone', 'ResponseName', 'outcome','Options',options);
% [codesSVM_FINAL, ~, ~, ~] = predict(trainedClassifier,features_main);
% [matSVM_FINAL,acc_SVM_FINAL,~] = createConfusionMatrix(codesTrue_main,codesSVM_FINAL);
% disp(acc_SVM_FINAL)

%% COMPILE RESULTS