%% Posterior Distribution Thresholding for Unknown Data
% Aakash Gupta (November 21, 2015)

%% LOAD DATA AND INITIALIZE PARAMETERS
clear all, close all;

p = gcp('nocreate');
if isempty(p)
    parpool('local')
end

cd(fileparts(which('unknownGENERATED.m')))
currentDir = pwd;
slashdir = '/';
addpath([pwd slashdir 'sub']); %create path to helper scripts
cd ../
ARCode = pwd;
code_folder = [ARCode '/code/'];
unk_folder = [ARCode '/unknown_data/'];
DSA_folder = [ARCode '/unknown_data/GENERATED/'];
DSA_raw_folder = [ARCode '/unknown_data/GENERATED/RawData/'];
DSA_feat_folder = [ARCode '/unknown_data/GENERATED/Features/'];
addpath(unk_folder)
addpath(DSA_folder)
addpath(DSA_raw_folder)
addpath(DSA_feat_folder)
addpath(code_folder)

plotON = 1;                             %draw plots
drawplot.activities = 0;                % show % of each activity
drawplot.accuracy = 0;
drawplot.actvstime = 0;
drawplot.confmat = 0;

%Additional options
clipThresh = 0; %to be in training set, clips must have >X% of label

OOBVarImp = 'off';   %enable variable importance measurement

%The HMM Transition Matrix (A)
transitionFile = 'A_5ActivityNSS.xlsx';
A = xlsread(transitionFile);

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

TF_mat = cell(2,5);
P_all = zeros(length(features),length(uniqStates));

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
activity_acc_matrix = zeros(5,folds);

% codesRF = ones(length(statesTrue),1);
% sigma = input('Sigma: ');

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
    
    %TRAIN RF
    ntrees = 300;
    
    %cost matrix 
    CostM = 0.5*ones(5,5); CostM([1 7 13 19 25]) = 0; 
    CostM(2,:) = 5; CostM(2,2) = 0;    %increase cost of misclassifying stairs
    CostM(3,:) = 5; CostM(3,3) = 0;
    
    tic; %start timer
    
    fprintf('\n')
    disp(['RF Train - Fold '  num2str(k) '  #Samples Train = ' num2str(N_session - (ind_change(k+1)-ind_change(k))) '  #Samples Test = ' num2str((ind_change(k+1)-ind_change(k)))]);
    
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
    
    %Plot var importance
    if strcmp(OOBVarImp,'on')
        figure('name',['k-fold ' num2str(k)])
        barh(RFmodel.OOBPermutedVarDeltaError);
        set(gca,'ytick',1:length(cData.featureLabels),'yticklabel',cData.featureLabels)
        grid on;
    end
    
    timeRF = toc; %end timer
      
    %RF Prediction and RF class probabilities for ENTIRE dataset. This is
    %for initializing the HMM Emission matrix (P_RF(TrainingSet)) and for
    %computing the observations of the HMM (P_RF(TestSet))
    [codesRF,P_RF] = predict(RFmodel,features);
    codesRF = str2num(cell2mat(codesRF));
    statesRF = uniqStates(codesRF);
    P_all(testSet,:) = P_RF(testSet,:);
    
%     template = templateSVM('KernelFunction', 'gaussian', 'PolynomialOrder', [], 'KernelScale', sigma, 'BoxConstraint', 1, 'Standardize', true);
%     trainedClassifier = fitcecoc(features(TrainingSet,:), codesTrue(TrainingSet)', 'Learners', template, 'FitPosterior', 1, 'Coding', 'onevsone', 'PredictorNames', cData.featureLabels, 'ResponseName', 'outcome');
%     [codesRF, ~, ~, P_RF] = predict(trainedClassifier,features);
%     statesRF = uniqStates(codesRF);
%     P_all(testSet,:) = P_RF(testSet,:);

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
    disp(['Train Time RF = ' num2str(timeRF) ' s'])
    
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
    
    for ii = 1:length(uniqStates)
        cT = codesTrue(testSet);
        cRF = codesRF(testSet);
        
        idx = find(cRF == ii);
        P_RF_temp = P_RF(testSet,ii);
        P_RF_activity = P_RF_temp(idx);
        TP_ind = find(cRF(idx) == cT(idx)'); %find indices of true postive
        FP_ind = find(~(cRF(idx) == cT(idx)')); %find indices of false
        
        TF_mat{1,ii} = [TF_mat{1,ii}; P_RF_activity(TP_ind)]; 
        TF_mat{2,ii} = [TF_mat{2,ii}; P_RF_activity(FP_ind)];
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

toc

%% Posterior Distrubtions
figure('name','Posterior Distributions by Class')
thresh = zeros(length(uniqStates),1);
bins = 20;
cutoff = 5; %cutoff percentile in percentage
for ii = 1:length(uniqStates)
    id = find(codesRF == ii); %get chosen activities (max posteriors)
    thresh(ii) = prctile(max(P_all(id,:),[],2),cutoff); %find posterior threshold from cutoff
    
    subplot(1,5,ii)
    histogram(max(P_all(id,:),[],2),bins);
    title({['Posteriors for ' uniqStates{ii}],['Cutoff at ' num2str(cutoff) '%: ' num2str(thresh(ii))]})
    xlim([0 1])
    yL = get(gca,'YLim');
    line([thresh(ii) thresh(ii)],yL,'Color','r');
end

%% Train RF Classifier On All Lab Data
RFmodel_all = TreeBagger(ntrees,features,codesTrue','OOBVarImp',OOBVarImp,'Cost',CostM,'Options',opts_ag);
%trainedClassifier_all = fitcecoc(features, codesTrue', 'Learners', template, 'FitPosterior', 1, 'Coding', 'onevsone', 'PredictorNames', cData.featureLabels, 'ResponseName', 'outcome');

%% Import Known Data + Predict On + Threshold
fprintf('\n')
file_known = [ARCode '/features_patient/CBR01_SCO_p.mat'];
disp(['Known Data: ' file_known])
load(file_known)

%Remove non-wearing
nonwear_ind = strmatch('Not Wearing',features_data.activity_labels,'exact');
features_data.subject(nonwear_ind) = [];
features_data.features(nonwear_ind,:) = [];
features_data.activity_labels(nonwear_ind) = [];
features_data.wearing_labels(nonwear_ind) = [];
features_data.identifier(nonwear_ind) = [];
features_data.activity_fraction(nonwear_ind) = [];
features_data.times(nonwear_ind) = [];
features_data.sessionID(nonwear_ind) = [];
features_data.subjectID(nonwear_ind) = [];

fprintf('\n')
disp('Predicting on known data.')
[codesRF_kn,P_RF_kn] = predict(RFmodel_all,features_data.features);
%[codesRF_kn, ~, ~, P_RF_kn] = predict(trainedClassifier_all,features);


[M_kn, I_kn] = max(P_RF_kn,[],2);
n_total_kn = length(M_kn);
n_correct_kn = length(find(M_kn > thresh(I_kn))); %correct known is GREATER than thresh
acc_kn = n_correct_kn./n_total_kn;
disp(['Correctly classified as known: ' num2str(acc_kn)])

%% Import Unknown Data + Process Data
%Import unknown data
X_unk = [];
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

%% Predict on Unknown Data + Threshold
fprintf('\n')
disp('Predicting on unknown data.')
[codesRF_unk,P_RF_unk] = predict(RFmodel_all,X_unk);
%[codesRF_unk, ~, ~, P_RF_unk] = predict(trainedClassifier_all,features);

[M_unk, I_unk] = max(P_RF_unk,[],2);
n_total_unk = length(M_unk);
n_correct_unk = length(find(M_unk < thresh(I_unk))); %correct unknown is LESS than thresh
acc_unk = n_correct_unk./n_total_unk;
disp(['Correctly classified as unknown: ' num2str(acc_unk)])

%% Optimize Threshold Values
%Known Data
n_total_kn = length(M_kn);
it = [0:0.01:1];
kn_mat = zeros(length(it),length(thresh));
for ii = 1:length(thresh)
    %thresh_it = thresh;
    thresh_it = zeros(5,1);
    for tt = 1:length(it)
        thresh_it(ii) = it(tt);        
        %kn_mat(tt,ii) = (length(find(M_kn > thresh_it(I_kn))))./n_total_kn;
        kn_mat(tt,ii) = (length(find(I_kn == ii & M_kn > thresh_it(I_kn))))./length(find(I_kn == ii));
    end
end

%Unknown Data
n_total_unk = length(M_unk);
it = [0:0.01:1];
unk_mat = zeros(length(it),length(thresh));
for ii = 1:length(thresh)
    %thresh_it = thresh;
    thresh_it = zeros(5,1);    
    for tt = 1:length(it)
        thresh_it(ii) = it(tt);        
        %unk_mat(tt,ii) = (length(find(M_unk < thresh_it(I_unk))))./n_total_unk;
        unk_mat(tt,ii) = (length(find(I_unk == ii & M_unk < thresh_it(I_unk))))./length(find(I_unk == ii));
    end
end

%Plot Known and Unknown Optimizations
figure;
it_max = zeros(size(unk_mat,2),1);
for b = 1:size(unk_mat,2)
    subplot(1,5,b)
    hold on
    plot(it, kn_mat(:,b),'LineWidth',3);
    plot(it, unk_mat(:,b),'LineWidth',3);
    
    %Plot maximum for both
    total = kn_mat(:,b) + unk_mat(:,b);
    [~,it_ind] = max(total);
    yL_2 = [0 1];
    line([it(it_ind) it(it_ind)],yL_2,'Color','r');
    it_max(b) = it(it_ind);
    
    %Find and plot intersection of known and unknown graphs
%     for cc = 2:length(kn_mat(:,b))-1
%         if ((unk_mat(cc-1,b) < kn_mat(cc,b)) && (unk_mat(cc+1,b) > kn_mat(cc,b))) || (round(unk_mat(cc,b),2) == round(kn_mat(cc,b),2))
%             it_max(b) = it(cc);
%             break
%         end
%     end
%     yL_3 = [0 1];
%     line([it(cc) it(cc)],yL_3,'Color','b');
%     it_max(b) = it(cc);
    
    legend('Known','Unknown')
    xlabel('Threshold Value','FontSize',18)
    ylabel('Accuracy','FontSize',18)
    title({StateCodes{b,1},['Known: ' num2str(100*length(find(I_kn == b))./n_total_kn) '%'],['Unknown: ' num2str(100*length(find(I_unk == b))./n_total_unk) '%']},'FontSize',20)
    set(gca,'Box','off','TickDir','out','LineWidth',2,'FontSize',14,'FontWeight','bold','XGrid','on');
    xlim([0 1])
    ylim([0 1])
    hold off
end

%% Generate Confusion Matrix for Optimized Threshold
mat_kn_unk = zeros(2,2);
mat_kn_unk(1,1) = length(find(M_kn > it_max(I_kn)))./n_total_kn;
mat_kn_unk(2,2) = length(find(M_unk < it_max(I_unk)))./n_total_unk;
mat_kn_unk(1,2) = 1 - mat_kn_unk(1,1);
mat_kn_unk(2,1) = 1 - mat_kn_unk(2,2);

figure;
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