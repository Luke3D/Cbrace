%% NEW SVM-HMM CODE for Train/Test - 1 Location Only (Belt)
%Rev by Luca Lonini 11.18.2013
%ver2: Save SVM and HMM models and accuracies
%ver3: Shows how many clips of each activity type we are removing
%Init HMM Emission Prob with PTrain from the SVM
%and Run the SVM+HMM on ALL data
%ver TIME: grab data for folds sequentially over time

%% LOAD DATA AND INITIALIZE PARAMETERS
clear all, close all;
addpath([pwd '/multisvmE']);
warning('off','all');
slashdir = '/';

currentDir = pwd; 
addpath([pwd slashdir 'sub']); %create path to helper scripts
addpath(genpath([slashdir 'Traindata'])); %add path for train data

plotON = 0;                             %draw plots
drawplot.activities = 0;                % show % of each activity
drawplot.accuracy = 0;
drawplot.actvstime = 0;
drawplot.confmat = 0;

%Additional options
clipThresh = 0; %to be in training set, clips must have >X% of label

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

%Scale each feature between 0 and 1 for SVM
cData = scaleFeaturesAG(cData);

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

%% SORT THE DATA FOR K-FOLDS + SVM TRAIN/TEST

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

%SessionID index ranges (differentiate CBR session 1 from SCO session 1)
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

%Optimize SVM rbf with sigma values:
sigma = [1 5 10 15 20 25 30 35 40 45 50 55 60 65 70 75 80 85 90 95 100];
optimization = zeros(length(user_subjects)+3,length(sigma));
optimization(1,:) = sigma;

%Do k fold cross validation for SVM
for ttt = 1:length(sigma)
    sigma(ttt)
    
    for k = 1:folds
        %% Create Train and Test vector - Split dataset into k-folds
        testSet = zeros(length(statesTrue),1);
        testSet(ind_change(k):ind_change(k+1)) = 1;
        testSet = logical(testSet);

        %Remove clips that are a mix of activities from training set
        %These were the clips that did not meet the 80% threshold
        TrainingSet = ~testSet;
        TrainingSet(removeInd) = 0;   %remove clips

        %% SVM TRAINING AND TESTING
        tic; %start timer

        fprintf('\n')
        disp(['SVM Train - Subject '  num2str(user_subjects(k)) '  #Samples Train = ' num2str(N_session - (ind_change(k+1)-ind_change(k))) '  #Samples Test = ' num2str((ind_change(k+1)-ind_change(k)))]);

        %How many samples of each activity we have in the test fold
        for i = 1:length(uniqStates)
            inda = find(strcmp(cData.activity(testSet),uniqStates(i)));
            Nsa = length(inda);
            indtot = find(strcmp(trainingClassifierData.activity,uniqStates(i)));
            Nsaperc = Nsa/length(indtot)*100;
            disp([num2str(Nsa) ' Samples of ' uniqStates{i} ' for this subject  (' num2str(Nsaperc) '% of total)'])
        end

        [trainresult,testresult] = multisvmE(features(TrainingSet,[1:14 16:29 31:44 46:end]),codesTrue(TrainingSet)',features(testSet,[1:14 16:29 31:44 46:end]),'rbf',sigma(ttt));

        timeSVM = toc; %end timer

        codesSVM = testresult;
        statesSVM = uniqStates(codesSVM);

        %% RESULTS for each k-fold
        [matSVM,accSVM,labels] = createConfusionMatrix(codesTrue(testSet),codesSVM);

        %Store all results
        results(k).stateCodes        = StateCodes;
        %OVERALL ACCURACY
        results(k).accSVM             = accSVM;
        %CONFUSION MATRICES
        results(k).matSVM             = matSVM;
        %PRED and ACTUAL CODES AND STATES
        results(k).statesSVM         = statesSVM;
        results(k).statesTrue       = statesTrue;
        results(k).trainingSet      = TrainingSet;
        results(k).testSet          = testSet;

        disp(['accSVM = ' num2str(accSVM)]);
        disp(['Train Time SVM = ' num2str(timeSVM) ' s'])

        %% PLOT PREDICTED AND ACTUAL STATES

        %Rename states for plot efficiency
        StateCodes(:,1) = {'Sit','Stairs Dw','Stairs Up','Stand','Walk'};
        nstates = length(StateCodes);

        if plotON
            if drawplot.actvstime
                dt = cData.clipdur * (1-cData.clipoverlap);
                t = 0:dt:dt*(length(codesTrue(testSet))-1);  
                figure('name',['k-fold ' num2str(k)]);
                plot(t,codesTrue(testSet),'.-g',t,codesSVM+.1,'.-r')
                xlabel('Time [s]')
                legend('True','SVM')
                ylim([0.5 nstates+0.5]);
                set(gca,'YTick',cell2mat(StateCodes(:,2))')
                set(gca,'YTickLabel',StateCodes(:,1))
            end

            if drawplot.accuracy
                figure;
                bar([accSVM]);
                if accSVM > 0.85
                    ylim([0.8 1])
                end
                set(gca,'XTickLabel',{'SVM'})
            end

            if drawplot.confmat
                figure('name',['k-fold ' num2str(k)]);
                correctones = sum(matSVM,2);
                correctones = repmat(correctones,[1 size(StateCodes,1)]);
                imagesc(matSVM./correctones); colorbar
                [cmin,cmax] = caxis;
                caxis([0,1])
                ax = gca;
                ax.XTick = 1:size(StateCodes,1);
                ax.YTick = 1:size(StateCodes,1);
                set(gca,'XTickLabel',{})
                set(gca,'YTickLabel',StateCodes(:,1))
                axis square

                activity_acc_matrix(:,k) = diag(matSVM./correctones);
            end
        end
    end

    %Display % of each activity over all predictions
    if drawplot.activities
        Activity = uniqStates;
        for a = 1:size(StateCodes,1)
            ind = [];  %temp var to store activity found
            for k = 1:folds
                ind = [ind; strcmp(results(k).statesSVM,Activity{a})];
            end
            Activity{a,2} = sum(ind)./size(ind,1)*100;  % the % of each activity 
        end 
        figure('name','% of activities')
        pchart = pie(cell2mat(Activity(:,2)),StateCodes(:,1));
    end

    %Average accuracy over all folds (weighted and unweighted)
    accSVM = 0; matSVM = zeros(nstates); matHmm = zeros(nstates);
    accSVM_uw = 0;
    for i = 1:folds
        accSVM = accSVM + results(i).accSVM.*(ind_change(i+1)-ind_change(i)); %weighted average
        correctones = sum(results(i).matSVM,2);
        correctones = repmat(correctones,[1 size(StateCodes,1)]);
        matSVM = matSVM + (results(i).matSVM)./correctones;
    end
    accSVM_uw = sum([results(:).accSVM])/folds;
    accSVM = accSVM/N_session;
    fprintf('\n')
    disp(['Unweighted Mean (k-Subject) accSVM = ' num2str(accSVM_uw)]);
    disp(['Weighted Mean (k-Subject) accSVM = ' num2str(accSVM)]);

    matSVM = matSVM./folds;
    matHmm = matHmm./folds;
    figure('name','Mean (k-fold) Confusion matrix')
    imagesc(matSVM); 
    colorbar
    [cmin,cmax] = caxis;
    caxis([0,1])
    ax = gca;
    ax.XTick = 1:size(StateCodes,1);
    ax.YTick = 1:size(StateCodes,1);
    set(gca,'XTickLabel',{})
    set(gca,'YTickLabel',StateCodes(:,1))
    axis square

    %% SAVE TRAINED MODELS
    % Results.results = results;
    % Results.SVMmodel = SVMmodel;
    % filename = ['Results_' cData.subject{1}];
    % save(filename,'Results');

    toc

    %% Accuracy Table
    test_size = zeros(folds+2,1);
    train_size = zeros(folds+2,1);
    row_folds = cell(folds+2,1);
    for ii = 1:folds
        test_size(ii) = (ind_change(ii+1)-ind_change(ii));
        train_size(ii) = N_session - (ind_change(ii+1)-ind_change(ii));
        row_folds(ii) = {['Subject ' num2str(user_subjects(ii))]};
    end

    row_folds(end-1) = {'Mean (UW)'};
    row_folds(end) =  {'Mean (W)'};

    acc_SVM_vector = [results.accSVM accSVM_uw accSVM]';

    acc_tbl = table(test_size,train_size,acc_SVM_vector,'RowNames',row_folds,'VariableNames',{'Test_Size','Train_Size','SVM_Acc'});
    fprintf('\n')
    disp(acc_tbl)
    
    optimization(2:end,ttt) = acc_SVM_vector;
end

warning('on','all')

figure
for zz = 1:length(user_subjects)
    hold on
    set(gca,'XScale','log','XGrid','on');
    semilogx(optimization(1,:),optimization(zz+1,:));
end
semilogx(optimization(1,:),optimization(end,:),'-','LineWidth',3);
xlabel('RBF Kernal Sigma Value')
ylabel('Accuracy')
hold off