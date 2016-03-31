%Test approach to determine unknown class
%Use hard threshold on RF posterior
%Train on 4 classes, use the 5th as the unknown

%% LOAD DATA AND INITIALIZE PARAMETERS
clear all, close all;
slashdir = '/';

currentDir = pwd;
addpath([pwd slashdir 'sub']); %create path to helper scripts
addpath(genpath([slashdir 'Traindata'])); %add path for train data

folds = 5;                              %number of folds to perform
plotON = 1;                             %draw plots
drawplot.activities = 1;                % show % of each activity
drawplot.accuracy = 0;
drawplot.actvstime = 1;
drawplot.confmat = 1;
drawplot.posterior = 1;
drawplot.posteriorhist = 1;

%Additional options
clipThresh = 0; %to be in training set, clips must have >X% of label

OOBVarImp = 'off';   %enable variable importance measurement

%The HMM Transition Matrix (A)
transitionFile = 'A_5ActivityNSS.xlsx';
A = xlsread(transitionFile);

%The unknown class
uc = 2; %1=sit, 2=dw, 3=up, 4=stand, 5=walk
A(uc,:) = []; A(:,uc) = []; %remove the unknown activity from A

PosteriorallTP = [];              %store the posterior of the classifier sorted by activity

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
fprintf('Available subjects to analyze: %s',tt)
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

%% FILTER DATA FOR APPROPRIATE CLASSIFICATION RUNTYPE

%Remove data from other locations if required (old datasets)
cData = removeDataWithoutLocation(cData,'Belt');

%rename the unknown state as X
iukw = strcmp(cData.activity,'Stairs Dw');
cData.activity(iukw) = repmat({'X'},[sum(iukw) 1]);

%create structure for test data and include the unknown class data
cDataTest = cData;
cDataTest.activity(~iukw) = [];
cDataTest.features(~iukw,:) = [];
cDataTest.activityFrac(~iukw) = [];

%remove unknown class from train data
cData.activity(iukw) = [];
cData.features(iukw,:) = [];
cData.activityFrac(iukw) = [];

%Split remaining train data into CV Data (4 classes) and Test Data (4 classes + Unknown
%class) - Use 70-30% split
indsplit = round(0.8*length(cData.activity));
%TEST DATA
cDataTest.activity = [cDataTest.activity ; cData.activity(indsplit:end)];
cDataTest.features = [cDataTest.features ; cData.features(indsplit:end,:)];
cDataTest.activityFrac = [cDataTest.activityFrac ; cData.activityFrac(indsplit:end)];
%CV DATA (Used for cross-validation)
cData.activity(indsplit:end) = [];
cData.features(indsplit:end,:) = [];
cData.activityFrac(indsplit:end) = [];

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

activity_acc_matrix = zeros(4,folds);

%Do k fold cross validation for RF
for k = 1:folds
    %% Create Train and Test vector - Split dataset into k-folds
    testSet  = zeros(length(statesTrue),1);
    Nsamples = floor(length(statesTrue)/folds);   %# of samples in each fold
    testSet((k-1)*Nsamples+1:k*Nsamples) = 1;
    testSet = logical(testSet);
    
    %Remove clips that are a mix of activities from training set
    %These were the clips that did not meet the 80% threshold
    TrainingSet = ~testSet;
    TrainingSet(removeInd) = 0;   %remove clips
    
    
    %% RF TRAINING AND TESTING
    
    %TRAIN RF
    ntrees = 100;
    
    
    tic; %start timer
    
    fprintf('\n')
    disp(['RF Train - Fold '  num2str(k) '  #Samples Train = ' num2str(Nsamples*(folds-1)) '  #Samples Test = ' num2str(Nsamples)]);
    
    %How many samples of each activity we have in the test fold
    for i = 1:length(uniqStates)
        %        inda = find(strcmp(trainingClassifierData.activity(testSet),uniqStates(i)));
        inda = find(strcmp(cData.activity(testSet),uniqStates(i)));
        Nsa = length(inda);
        indtot = find(strcmp(trainingClassifierData.activity,uniqStates(i)));
        Nsaperc = Nsa/length(indtot)*100;
        disp([num2str(Nsa) ' Samples of ' uniqStates{i} ' in this fold  (' num2str(Nsaperc) '% of total)'])
    end
    
    RFmodel = TreeBagger(ntrees,features(TrainingSet,:),codesTrue(TrainingSet)','OOBVarImp',OOBVarImp);
    
    %Plot var importance
    if strcmp(OOBVarImp,'on')
        figure('name',['k-fold ' num2str(k)])
        barh(RFmodel.OOBPermutedVarDeltaError);
        set(gca,'ytick',1:length(cData.featureLabels),'yticklabel',cData.featureLabels)
        grid on;
    end
    
    timeRF = toc; %end timer
    
    %RF Prediction and RF class probabilities for ENTIRE dataset. This is
    %for initializing the HMM Emission matrix (P_RF(TrainSet)) and for
    %computing the observations of the HMM (P_RF(TestSet))
    [codesRF,P_RF] = predict(RFmodel,features);
    codesRF = str2num(cell2mat(codesRF));
    statesRF = uniqStates(codesRF);
    
    
    %% HMM INIT AND TESTING
    
    %Initialize parameters for HMM
    d       = length(uniqStates);   %number of symbols (=#states)
    nstates = d;                    %number of states
    mu      = zeros(d,nstates);     %mean of emission distribution
    sigma   = zeros(d,1,nstates);   %std dev of emission distribution
    Pi      = ones(length(uniqStates),1) ./ length(uniqStates); %uniform prior
    sigmaC  = .1;                   %use a constant std dev
    PTrain = P_RF(TrainingSet,:);   %Emission prob = RF class prob for training data (! <80% threshold data is excluded)
    
    %Create emission probabilities for HMM
    PBins  = cell(d,1);
    
    %For each type of state we need a distribution
    for bin = 1:d
        clipInd         = strcmp(uniqStates{bin},statesTrue(TrainingSet));
        PBins{bin,1}    = PTrain(clipInd,:);    %The emission probability of the HMM is the RF prob over training data
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
    
    %% Predict the states on Testdata using the HMM
    disp(['HMM Train - fold '  num2str(k)]);
    
    PTest = P_RF(testSet,:);        %The observation sequence (Test data)
    [gamma, ~, ~, ~, ~]   = hmmInferNodes(HMMmodel,PTest');
    [statesHmm, codesHmm] = getPredictedStates(gamma',uniqStates);
    
    timeHMM = toc;
    
    %% RESULTS for each k-fold
    %entire classification matrix (HMM prediction is run only on Test data)
    [matRF,accRF,labels] = createConfusionMatrix(codesTrue(testSet),codesRF(testSet));
    [matHmm,accHmm,labels] = createConfusionMatrix(codesTrue(testSet),codesHmm);
    
    %Store all results
    results(k).stateCodes        = StateCodes;
    %OVERALL ACCURACY
    results(k).accRF             = accRF;
    results(k).accHmm            = accHmm;
    %CONFUSION MATRICES
    results(k).matRF             = matRF;
    results(k).matHmm            = matHmm;
    %PRED and ACTUAL CODES AND STATES
    results(k).treepred         = PTest;  %RF output on test dataset
    results(k).statesRF         = statesRF(testSet);
    results(k).statesHmm        = statesHmm;
    results(k).statesTrue       = statesTrue(testSet);
    results(k).trainingSet      = TrainingSet;
    results(k).testSet          = testSet;
    %     results(k).codesTrue       = codesTrue(testSet);
    %     results(k).codesRF         = codesRF(testSet);
    %     results(k).codesHmm        = codesHmm;
    
    disp(['accRF = ' num2str(accRF)]);
    disp(['accHmm = ' num2str(accHmm)]);
    disp(['Train Time RF = ' num2str(timeRF) ' s'])
    disp(['Train Time RF+HMM = ' num2str(timeHMM) ' s'])
    
    %% PLOT PREDICTED AND ACTUAL STATES
    
    %Rename states for plot efficiency
    %     StateCodes(:,1) = {'Sit','Stairs Dw','Stairs Up','Stand','Walk'};
    
    if plotON
        if drawplot.actvstime
            dt = cData.clipdur * (1-cData.clipoverlap);
            t = 0:dt:dt*(length(codesTrue(testSet))-1);
            figure('name',['k-fold ' num2str(k)]); hold on
            subplot(211), hold on
            plot(t,codesTrue(testSet),'.-g')
            plot(t,codesRF(testSet)+.1,'.-r')
            plot(t,codesHmm+.2,'.-b')
            xlabel('Time [s]')
            legend('True','RF','HMM')
            ylim([0.5 nstates+0.5]);
            set(gca,'YTick',cell2mat(StateCodes(:,2))')
            set(gca,'YTickLabel',StateCodes(:,1))
            subplot(212)
            plot(t,max(P_RF(testSet,:),[],2),'r'), hold on
            line([0 t(end)],[1/nstates 1/nstates])
        end
        
        if drawplot.accuracy
            
            figure;
            bar([accRF accHmm]);
            if accRF > 0.85 && accHmm > 0.85
                ylim([0.8 1])
            end
            set(gca,'XTickLabel',{'RF','RF+HMM'})
        end
        
        if drawplot.confmat
            figure('name',['k-fold ' num2str(k)]); hold on
            correctones = sum(matRF,2);
            correctones = repmat(correctones,[1 size(StateCodes,1)]);
            subplot(121); imagesc(matRF./correctones); colorbar
            set(gca,'XTickLabel',StateCodes(:,1))
            set(gca,'YTickLabel',StateCodes(:,1))
            axis square
            subplot(122); imagesc(matHmm./correctones); colorbar
            set(gca,'XTickLabel',StateCodes(:,1))
            set(gca,'YTickLabel',StateCodes(:,1))
            axis square
            
            activity_acc_matrix(:,k) = diag(matRF./correctones);
        end
    end
    
    if drawplot.posterior
        PosterioraTP = [];
        Posterior = P_RF(testSet,:);    %posterior for each class for data in fold k
        codesRFTest = codesRF(testSet);  %predicted activities in fold k
        codesTrueTest = codesTrue(testSet)';  %true activities in fold k
        for a = 1:length(uniqStates)
            idx = find(codesRFTest == a);  %predicted activity a
            Posteriora = Posterior(idx,a);   %posterior for activity a (TP and FP)
            TP_ind = find(codesRFTest(idx) == codesTrueTest(idx)); %find indices of true postive
            PosterioraTP = [PosterioraTP; Posteriora(TP_ind) repmat(a,size(TP_ind))];    %posterior for TP only, 2nd col is activity
            %             PosteriorallTP = [PosteriorallTP; Posteriora(TP_ind) repmat(a,size(TP_ind))]; %store posterior for all folds
        end
        figure('name',['k-fold ' num2str(k)]);
        boxplot(PosterioraTP(1:end,1),PosterioraTP(:,end),'labels',StateCodes(:,1))
        
        %the posterior across all folds
        PosteriorallTP = [PosteriorallTP; PosterioraTP]; %store posterior for all folds
        if k == folds
            figure('name','Posterior for all TP');
            boxplot(PosteriorallTP(1:end,1),PosteriorallTP(:,end),'labels',StateCodes(:,1))
        end
        %the 25th percentile per activity
        for a = 1:length(uniqStates)
            thrprc(a) = prctile(PosteriorallTP(PosteriorallTP(:,2)==a),25);
        end
        
        if drawplot.posteriorhist && k == folds
            figure('name','Histogram of Posterior for TP')
            for a = 1:length(uniqStates)
                subplot(1,5,a)
                histogram(PosteriorallTP(PosteriorallTP(:,2)==a))
            end
            
        end
        
        
    end
    
end

%Display % of each activity over all predictions
if drawplot.activities
    Activity = uniqStates;
    for a = 1:size(StateCodes,1)
        ind = [];  %temp var to store activity found
        for k = 1:folds
            ind = [ind; strcmp(results(k).statesHmm,Activity{a})];
        end
        Activity{a,2} = sum(ind)./size(ind,1)*100;  % the % of each activity
    end
    figure('name','% of activities')
    pchart = pie(cell2mat(Activity(:,2)),StateCodes(:,1));
    %     bar(cell2mat(Activity(:,2)));
    %     set(gca,'XTickLabel',StateCodes(:,1))
    %     ylabel('% of time spent')
end

%Average accuracy over all folds
accRF = 0; accHmm = 0; matRF = zeros(nstates); matHmm = zeros(nstates);
for i = 1:folds
    accRF = accRF + results(i).accRF;
    accHmm = accHmm + results(i).accHmm;
    correctones = sum(results(i).matRF,2);
    correctones = repmat(correctones,[1 size(StateCodes,1)]);
    matRF = matRF + (results(i).matRF)./correctones;
    matHmm = matHmm + (results(i).matHmm)./correctones;
end
accRF = accRF/folds;
accHmm = accHmm/folds;
fprintf('\n')
disp(['Mean (k-fold) accRF = ' num2str(accRF)]);
disp(['Mean (k-fold) accHmm = ' num2str(accHmm)]);

matRF = matRF./folds;
matHmm = matHmm./folds;
figure('name','Mean (k-fold) Confusion matrix')
subplot(121); imagesc(matRF);
colorbar
[cmin,cmax] = caxis;
caxis([0,1])
ax = gca;
ax.XTick = 1:size(StateCodes,1);
ax.YTick = 1:size(StateCodes,1);
set(gca,'XTickLabel',{})
set(gca,'YTickLabel',StateCodes(:,1))
axis square
subplot(122); imagesc(matHmm);
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
% Results.RFmodel = RFmodel;
% Results.HMMmodel = HMMmodel;
% filename = ['Results_' cData.subject{1}];
% save(filename,'Results');

toc

%% Accuracy Table
test_size = zeros(folds+1,1);
train_size = zeros(folds+1,1);
row_folds = cell(folds+1,1);

test_size(1:folds) = Nsamples;
train_size(1:folds) = Nsamples*(folds-1);

for ii = 1:folds
    row_folds(ii) = {['Fold ' num2str(ii)]};
end

row_folds(end) =  {'Mean'};

acc_RF_vector = [results.accRF accRF]';
acc_Hmm_vector = [results.accHmm accHmm]';

acc_tbl = table(test_size,train_size,acc_RF_vector,acc_Hmm_vector,'RowNames',row_folds,'VariableNames',{'Test_Size','Train_Size','RF_Acc','Hmm_Acc'});
fprintf('\n')
disp(acc_tbl)

%% Activity Accuracy Table
temp2 = mean(activity_acc_matrix');
activity_acc = [activity_acc_matrix temp2'];
% act = {'Sitting';'Stairs Dw';'Stairs Up';'Standing';'Walking'};
var_name = cell(folds+1,1);
for ii = 1:folds
    var_name(ii) = {['Fold_' num2str(ii)]};
end
var_name(end) = {'Mean'};

activity_tbl = array2table(activity_acc,'RowNames',StateCodes(:,1),'VariableNames',var_name);

%% Now train a forest on all CV data and test on the unknown class (Test Set)

Xtrain = cData.features;
Ytrain = codesTrue';

Xtest = cDataTest.features; %contains the unknown class
Ytest = cDataTest.activity;
Ytest = double(categorical(Ytest)); %convert to codes
StateCodes{5,1}='X';  StateCodes{5,2}=5;  %the codes

RFmodel = TreeBagger(ntrees,Xtrain,Ytrain);
[codesRFTest,P_RFTest] = predict(RFmodel,Xtest);
codesRFTest = str2double(codesRFTest)
%filter the prediction with threshold
P_RFTest = max(P_RFTest,[],2);
P_RFTest = [P_RFTest codesRFTest];
P_RFTest = [P_RFTest thrprc(codesRFTest)'];

iX = find(P_RFTest(:,1) < P_RFTest(:,3));
codesRFTest(iX) = 5;    %assign the unknown class to Posteriors below the threshold

cmat = confusionmat(Ytest,codesRFTest)
acc = trace(cmat)/sum(sum(cmat));   %accuracy
correctones = sum(cmat,2);
correctones = repmat(correctones,[1 length(unique(Ytest))]);
figure
imagesc(cmat./correctones); colorbar
colorbar
[cmin,cmax] = caxis;
caxis([0,1])
ax = gca;
ax.XTick = 1:size(StateCodes,1);
ax.YTick = 1:size(StateCodes,1);
set(gca,'XTickLabel',{})
set(gca,'YTickLabel',StateCodes(:,1))
axis square
title(['Accuracy = ', num2str(acc)])

%need to plot known and unknown
