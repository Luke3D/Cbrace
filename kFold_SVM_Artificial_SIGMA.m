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

%% OPTIMIZE SIGMA VALUE
sigma = [1:0.25:15];
cData_for = cData;
optimization = zeros(length(user_subjects)+1,length(sigma));
optimization(1,:) = sigma;

for yyy = 1:length(user_subjects)
    user_subjects(yyy)
    cData = isolateSubject(cData_for,find(user_subjects(yyy)==cData_for.subjectID));
    
    for xxx = 1:length(sigma)
        sigma(xxx)
        
        %% FILTER DATA FOR APPROPRIATE CLASSIFICATION RUNTYPE

        %Remove data from other locations if required (old datasets)
        cData = removeDataWithoutLocation(cData,'Belt');

        %Rescale each feature between 0 and 1 for SVM
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
        N_session = length(cData.sessionID);
        ind_change = [1]; %include first index;
        for kk = 1:(N_session-1)
            if ~((cData.sessionID(kk+1))-(cData.sessionID(kk)) == 0)
                ind_change = [ind_change kk]; %store all the sessionID change indices
            end
        end
        ind_change = [ind_change N_session]; %include last index

        folds = length(ind_change) - 1; %number of folds = number of sessions

        %Sort Activities
        sitting_ind = strmatch('Sitting',cData.activity,'exact');
        walking_ind = strmatch('Walking',cData.activity,'exact');
        standing_ind = strmatch('Standing',cData.activity,'exact');
        stairs_up_ind = strmatch('Stairs Up',cData.activity,'exact');
        stairs_dw_ind = strmatch('Stairs Dw',cData.activity,'exact');

        sitting_1_interval = [1];
        sitting_2_interval = [1];
        sitting_3_interval = [1];
        walking_1_interval = [1];
        walking_2_interval = [1];
        standing_1_interval = [1];
        standing_2_interval = [1];
        standing_3_interval = [1];
        standing_4_interval = [1];
        standing_5_interval = [1];
        stairs_up_interval = [1];
        stairs_dw_interval = [1];

        N_sit = length(sitting_ind);
        N_walk = length(walking_ind);
        N_stand = length(standing_ind);

        sitting_ind_1 = sitting_ind(1:floor(N_sit/3));
        sitting_ind_2 = sitting_ind(floor(N_sit/3):2*(floor(N_sit/3)));
        sitting_ind_3 = sitting_ind(2*(floor(N_sit/3)):end);
        walking_ind_1 = walking_ind(1:floor(N_walk/2));
        walking_ind_2 = walking_ind(floor(N_walk/2):end);
        standing_ind_1 = standing_ind(1:floor(N_stand/5));
        standing_ind_2 = standing_ind(floor(N_stand/5):2*(floor(N_stand/5)));
        standing_ind_3 = standing_ind(2*(floor(N_stand/5)):3*(floor(N_stand/5)));
        standing_ind_4 = standing_ind(3*(floor(N_stand/5)):4*(floor(N_stand/5)));
        standing_ind_5 = standing_ind(4*(floor(N_stand/5)):end);


        for tt = 1:(folds-1)
            sitting_1_interval = [sitting_1_interval tt.*floor(length(sitting_ind_1)/folds)];
            sitting_2_interval = [sitting_2_interval tt.*floor(length(sitting_ind_2)/folds)];
            sitting_3_interval = [sitting_3_interval tt.*floor(length(sitting_ind_3)/folds)];

            walking_1_interval = [walking_1_interval tt.*floor(length(walking_ind_1)/folds)];
            walking_2_interval = [walking_2_interval tt.*floor(length(walking_ind_2)/folds)];

            standing_1_interval = [standing_1_interval tt.*floor(length(standing_ind_1)/folds)];
            standing_2_interval = [standing_2_interval tt.*floor(length(standing_ind_2)/folds)];
            standing_3_interval = [standing_3_interval tt.*floor(length(standing_ind_3)/folds)];
            standing_4_interval = [standing_4_interval tt.*floor(length(standing_ind_4)/folds)];
            standing_5_interval = [standing_5_interval tt.*floor(length(standing_ind_5)/folds)];

            stairs_up_interval = [stairs_up_interval tt.*floor(length(stairs_up_ind)/folds)];
            stairs_dw_interval = [stairs_dw_interval tt.*floor(length(stairs_dw_ind)/folds)];
        end

        sitting_1_interval = [sitting_1_interval length(sitting_ind_1)];
        sitting_2_interval = [sitting_2_interval length(sitting_ind_2)];
        sitting_3_interval = [sitting_3_interval length(sitting_ind_3)];
        walking_1_interval = [walking_1_interval length(walking_ind_1)];
        walking_2_interval = [walking_2_interval length(walking_ind_2)];
        standing_1_interval = [standing_1_interval length(standing_ind_1)];
        standing_2_interval = [standing_2_interval length(standing_ind_2)];
        standing_3_interval = [standing_3_interval length(standing_ind_3)];
        standing_4_interval = [standing_4_interval length(standing_ind_4)];
        standing_5_interval = [standing_5_interval length(standing_ind_5)];
        stairs_up_interval = [stairs_up_interval length(stairs_up_ind)];
        stairs_dw_interval = [stairs_dw_interval length(stairs_dw_ind)];

        test_size = zeros(folds,1);

        activity_acc_matrix = zeros(5,folds);

        %Do k fold cross validation for SVM
        for k = 1:folds
            %% Create Train and Test vector - Split dataset into k-folds
            testSet = zeros(length(statesTrue),1);
            %Sit-Stand-Sit-Stand-Walk-Stand-Stairs_Dw-Stand-Stairs_Up-Stand-Walk-Stand-Sit
            test_ind = [sitting_ind_1(sitting_1_interval(k):sitting_1_interval(k+1)); ...
                        standing_ind_1(standing_1_interval(k):standing_1_interval(k+1)); ...
                        sitting_ind_2(sitting_2_interval(k):sitting_2_interval(k+1)); ...
                        standing_ind_2(standing_2_interval(k):standing_2_interval(k+1)); ...
                        walking_ind_1(walking_1_interval(k):walking_1_interval(k+1)); ...
                        standing_ind_3(standing_3_interval(k):standing_3_interval(k+1)); ...
                        stairs_dw_ind(stairs_dw_interval(k):stairs_dw_interval(k+1)); ...
                        standing_ind_4(standing_4_interval(k):standing_4_interval(k+1)); ...
                        stairs_up_ind(stairs_up_interval(k):stairs_up_interval(k+1)); ...
                        walking_ind_2(walking_2_interval(k):walking_2_interval(k+1)); ...
                        standing_ind_5(standing_5_interval(k):standing_5_interval(k+1)); ...
                        sitting_ind_3(sitting_3_interval(k):sitting_3_interval(k+1))];
            test_size(k) = length(test_ind);
            testSet(test_ind) = 1;
            testSet = logical(testSet);

            %Remove clips that are a mix of activities from training set
            %These were the clips that did not meet the 80% threshold
            TrainingSet = ~testSet;
            TrainingSet(removeInd) = 0;   %remove clips

            %% SVM TRAINING AND TESTING

            tic; %start timer

            fprintf('\n')
            disp(['SVM Train - Fold '  num2str(k) '  #Samples Train = ' num2str(N_session - test_size(k)) '  #Samples Test = ' num2str(test_size(k))]);

            %How many samples of each activity we have in the test fold
            for i = 1:length(uniqStates)
                inda = find(strcmp(cData.activity(testSet),uniqStates(i)));
                Nsa = length(inda);
                indtot = find(strcmp(trainingClassifierData.activity,uniqStates(i)));
                Nsaperc = Nsa/length(indtot)*100;
                disp([num2str(Nsa) ' Samples of ' uniqStates{i} ' in this fold  (' num2str(Nsaperc) '% of total)'])
            end

            warning off;
            [trainresult,testresult] = multisvmE(features(TrainingSet,:),codesTrue(TrainingSet)',features(testSet,:),'rbf',sigma(xxx));
            warning on;

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
        %     bar(cell2mat(Activity(:,2)));
        %     set(gca,'XTickLabel',StateCodes(:,1))
        %     ylabel('% of time spent')
        end

        %Average accuracy over all folds (weighted and unweighted)
        accSVM = 0; matSVM = zeros(nstates); accSVM_uw = 0; 
        for i = 1:folds
            accSVM = accSVM + results(i).accSVM.*(test_size(i)); %weighted average
            correctones = sum(results(i).matSVM,2);
            correctones = repmat(correctones,[1 size(StateCodes,1)]);
            matSVM = matSVM + (results(i).matSVM)./correctones;
        end
        accSVM_uw = sum([results(:).accSVM])/folds;
        accSVM = accSVM/sum(test_size);
        fprintf('\n')
        disp(['Unweighted Mean (k-fold) accSVM = ' num2str(accSVM_uw)]);
        disp(['Weighted Mean (k-fold) accSVM = ' num2str(accSVM)]);

%         matSVM = matSVM./folds;
%         figure('name','Mean (k-fold) Confusion matrix')
%         imagesc(matSVM); 
%         colorbar
%         [cmin,cmax] = caxis;
%         caxis([0,1])
%         ax = gca;
%         ax.XTick = 1:size(StateCodes,1);
%         ax.YTick = 1:size(StateCodes,1);
%         set(gca,'XTickLabel',{})
%         set(gca,'YTickLabel',StateCodes(:,1))
%         axis square

        %% SAVE TRAINED MODELS
        % Results.results = results;
        % Results.SVMmodel = SVMmodel;
        % Results.HMMmodel = HMMmodel;
        % filename = ['Results_' cData.subject{1}];
        % save(filename,'Results');

        toc

        %% Accuracy Table
        test_size_2 = zeros(folds+2,1);
        train_size = zeros(folds+2,1);
        row_folds = cell(folds+2,1);
        for ii = 1:folds
            test_size_2(ii) = test_size(ii);
            train_size(ii) = N_session - test_size(ii);
            row_folds(ii) = {['Fold ' num2str(ii)]};
        end

        row_folds(end-1) = {'Mean (UW)'};
        row_folds(end) =  {'Mean (W)'};

        acc_SVM_vector = [results.accSVM accSVM_uw accSVM]';

        acc_tbl = table(test_size_2,train_size,acc_SVM_vector,'RowNames',row_folds,'VariableNames',{'Test_Size','Train_Size','SVM_Acc'});
        fprintf('\n')
        disp(acc_tbl)

%         %% Activity Accuracy Table
%         temp2 = mean(activity_acc_matrix');
%         activity_acc = [activity_acc_matrix temp2'];
%         act = {'Sitting';'Stairs Dw';'Stairs Up';'Standing';'Walking'};
%         var_name = cell(folds+1,1);
%         for ii = 1:folds
%             var_name(ii) = {['Fold_' num2str(ii)]};
%         end
%         var_name(end) = {'Mean'};
% 
%         activity_tbl = array2table(activity_acc,'RowNames',act,'VariableNames',var_name);
%         disp(activity_tbl)
        
        optimization(yyy+1,xxx) = accSVM;
    end
end

figure;
sigma_max = zeros(2,length(user_subjects));
sigma_max(1,:) = user_subjects;
for zz = 1:length(user_subjects)
    hold on
    set(gca,'XScale','log','XGrid','on');
    semilogx(optimization(1,:),optimization(zz+1,:));
    [M,I] = max(optimization(zz+1,:));
    sigma_max(2,zz) = optimization(1,I);
    plot(optimization(1,I),optimization(zz+1,I),'o')
    xlabel('RBF Kernel Sigma Value')
    ylabel('Accuracy')
    xlim([min(sigma) max(sigma)]);
end
hold off