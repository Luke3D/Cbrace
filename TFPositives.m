%True-Positives and False-Positives Distributions

%% LOAD NECESSARY DIRECTORIES
clear all; close all;
warning('off','all')
slashdir = '/';

cd(fileparts(which('TFPositives.m')))
currentDir = pwd;
addpath([pwd slashdir 'sub']); %create path to helper scripts
addpath(genpath([slashdir 'Traindata'])); %add path for train data

current = pwd;
ARCode = current(1:end-4);
home_labeled_folder = [ARCode 'home_labeled/'];
addpath(home_labeled_folder);
disp('Home Labeled Data folder loaded.');

ExtractFeatures = 0;                 %if all features need to be extracted
WearTime = 1;                           %if Wear time only are extracted
plotON = 1;                             %draw plots
drawplot.activities = 0;
drawplot.actvstime = 1;
drawplot.confmat = 1;

%% LOAD LAB TRAINING DATA + SELECT SUBJECT/BRACE TO ANALYZE

%Load .mat file with all patient lab training data
filename = 'trainData_patient.mat';
load(filename)
disp('Training data file loaded.')

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
    if ~(strcmpi(brace_analyze,'SCO') || strcmpi(brace_analyze,'CBR'))
        disp('---------------------------------------------------------')
        disp('Please correctly select a brace (SCO or CBR). Try again.');
        disp('---------------------------------------------------------')
    else
        %Check if SCO or CBR are in mat file
        if (strcmpi(brace_analyze,'CBR'))
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
cData = isolateBrace(cData_temp2,brace_analyze);

%Display which training file will be used
fprintf('\n')
disp('These are the subject files that will be analyzed: ')
disp(unique(cData.subject))

%Set up directories for this subject
if subject_analyze < 10
    subj_str = ['0' num2str(subject_analyze)];
elseif subject_analyze > 10
    subj_str = num2str(subject_analyze);
end
subj_home_folder = [home_labeled_folder 'CBR' subj_str '_' upper(brace_analyze) '/'];
addpath(subj_home_folder);
fprintf('\n')
disp('Subject home data folder loaded.')
dirList = dir(subj_home_folder);
dirFeat = dir(home_labeled_folder);

%Check if home labeled features file is present
exists = 0;
for z = 1:length(dirFeat)
    dirFiles = dirFeat(z).name;
    if strcmpi(dirFiles,['CBR' subj_str '_' upper(brace_analyze) '_HOME_FEAT.mat'])
        disp(['Home labeled features file present for CBR' subj_str ' - ' upper(brace_analyze) '.'])
        exists = 1;
    end
end

if ~exists %features do not already exist
    disp(['Home labeled features file is NOT present for CBR' subj_str ' - ' upper(brace_analyze) '.'])
    disp(['Home labeled features file will be generated.'])
end

%% IMPORT DATA + GENERATE CLIPS AND FEATURES
if ~exists
    p = gcp('nocreate');
    if isempty(p)
        parpool('local')
    end
    
    N_files = length({dirList(:).name});
    Clipdur = cData.clipdur;                  %clip length in secs
    Nfeat = size(cData.features,2);       %number of features
    Sf = 30;                                  %the sampling freq [Hz]
    dur = Clipdur*Sf;    %the clip length [samples]
%    cData.clipoverlap = 0; %SET CLIP OVERLAP TO 0 (REMOVE THIS LINE FOR DEFAULT OVERLAP)
    dur_overlap = dur - dur*cData.clipoverlap; %nonoverlapping duration for clip (1.5 seconds - 45 data points)
    Xall = []; codesTrue_home = []; statesTrue_home = {}; size_raw = 0;
    activities_5 = {'Sitting','Stairs Dw','Stairs Up','Standing','Walking'};
    
    file_ind = [1:N_files];
    remove_ind = [];
    start = 0;
    for ii = 1:N_files
        temp = dirList(ii).name;
        if temp(1) == '.' %hidden filess
            start = start + 1;
            remove_ind(ii) = ii;
        end
    end
    file_ind(remove_ind) = [];
    
    fprintf('\n')
    N_home_files = length(file_ind);
    disp([num2str(N_home_files) ' home labeled data files were located.'])
    
    tic
    for k = file_ind
        clearvars ds
        datafile = [subj_home_folder dirList(k).name];
        ds = datastore(datafile,'ReadVariableNames',0);
        ds.RowsPerRead = 360000; %integer number of clips
        disp(['File ' num2str(k-start) ' loaded.']);
        
        while hasdata(ds)
            imp = read(ds);
            acc = table2array(imp(:,{'Var2','Var3','Var4'}));
            labels_activity = imp{:,{'Var7'}};
            
            %Remove non-wearing data:
            nonwear_ind = [strmatch('Not Wearing',labels_activity,'exact'); strmatch('Stand to Sit',labels_activity,'exact'); strmatch('Sit to Stand',labels_activity,'exact')];
            labels_activity(nonwear_ind) = [];
            acc(nonwear_ind,:) = [];
            
            up_ind = strmatch('Stairs Up',labels_activity,'exact');
            down_ind = strmatch('Stairs Dw',labels_activity,'exact');
            labels_activity(up_ind) = {'Stairs Dw'};
            labels_activity(down_ind) = {'Stairs Up'};
            
            size_raw = size_raw + length(acc);
            
            %Extract clips and generate features for clips
            Nclips = floor(size(acc,1)/dur_overlap - 3);
            %X_feat = zeros(Nclips,Nfeat);    %matrix with features for all clips
            X_feat = [];
            stateCodes = zeros(1,Nclips);
            states = cell(Nclips,1);
            parfor c = 1:Nclips
                wind = [((c-1)*dur_overlap+1):((c-1)*dur_overlap+1+dur)];
                X_feat(c,:) = getFeaturesHOME(acc(wind,:)'); %generate features
                
                %Find majority of activity in 6 second clip
                activity_tally = zeros(1,5);
                labels_cropped = labels_activity(wind);
                for ii = 1:dur
                    tally = strmatch(labels_cropped(ii),activities_5,'exact');
                    activity_tally(tally) = activity_tally(tally) + 1;
                end
                [~,I] = max(activity_tally);
                stateCodes(c) = I;
                states(c) = activities_5(I);
            end
            
            Xall = [Xall; X_feat];
            codesTrue_home = [codesTrue_home stateCodes]; 
            statesTrue_home = [statesTrue_home; states];
        end

        disp(['Features generated for file ' num2str(k-start) '.'])
    end
    
    tel = toc;
    fprintf('\n')
    disp(['Feature generation completed in ' num2str(tel) ' seconds.'])
    
    %Saving features file
    fprintf('\n')
    disp(['Saving features file...'])
    filename_save = ['CBR' subj_str '_' upper(brace_analyze) '_HOME_FEAT.mat'];
    save([home_labeled_folder filename_save],'Xall','codesTrue_home','statesTrue_home','size_raw')
    disp('File saved.')
    fprintf('\n')
else
    load(['CBR' subj_str '_' upper(brace_analyze) '_HOME_FEAT.mat'])
end

%% PREPARE FOR TRAINING
%The HMM Transition Matrix (A)
transitionFile = 'A_5ActivityNSS.xlsx';
A = xlsread(transitionFile);

%Clip threshold options
clipThresh = 0; %to be in training set, clips must have >X% of label

% cData = scaleFeatures(cData); %scale to [0 1]
cData = cData;

%remove data from other locations if required (old datasets)
cData = removeDataWithoutLocation(cData,'Belt');

%create local variables for often used data
features     = cData.features; %features for classifier
subjects     = cData.subject;  %subject number
uniqSubjects = unique(subjects); %list of subjects
statesTrue   = cData.activity;     %all the classifier data
uniqStates   = unique(statesTrue_home);  %set of states we have

%How many clips of each activity type we removed
if clipThresh > 0
    %remove any clips that don't meet the training set threshold
    [cData, removeInd] = removeDataWithActivityFraction(cData,clipThresh);

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

%Store Code and label of each unique State
StateCodes = cell(length(uniqStates),2);
StateCodes(:,1) = uniqStates;
StateCodes(:,2) = num2cell(1:length(uniqStates)); %sorted by unique

%% TRAIN RF (standard parameters and save results)
ntrees = 300;
fprintf('\n')
disp(['RF Train - Number of samples train = ' num2str(size(features,1))])

RFmodel = TreeBagger(ntrees,features,codesTrue');
[~,P_TrainRF] = predict(RFmodel,features);

%RF Prediction and RF class probabilities for ENTIRE dataset. This is
%for initializing the HMM Emission matrix (P_RF(TrainSet)) and for
%computing the observations of the HMM (P_RF(TestSet))
[codesRF,P_RF] = predict(RFmodel,Xall);
codesRF = str2num(cell2mat(codesRF));
statesRF = uniqStates(codesRF);

disp('RF Model trained.')

%% TRAIN HMM (i.e. create HMM and set the emission prob as the RF output)
PTrain = P_TrainRF;  %The Emission Probabilities of the HMM are the RF output prob on the train dataset

%Initialize parameters for hmm
d       = length(uniqStates);   %number of symbols (=#states)
nstates = d;                    %number of states
mu      = zeros(d,nstates);     %mean of emission distribution
sigma   = zeros(d,1,nstates);   %std dev of emission distribution
Pi      = ones(length(uniqStates),1) ./ length(uniqStates); %uniform prior
sigmaC  = .1;                   %use a constant std dev

%Create emission probabilities for HMM
PBins  = cell(d,1);

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

PTest = P_RF;        %The observation sequence (Test data)
[gamma, ~, ~, ~, ~]   = hmmInferNodes(HMMmodel,PTest');
[statesHmm, codesHmm] = getPredictedStates(gamma',uniqStates);

%Save RF and HMM models
disp('Saving RF and HMM models...')
file_models = ['CBR' subj_str '_' upper(brace_analyze) '_HOME_MODELS.mat'];
save([home_labeled_folder file_models],'RFmodel')
disp('RF and HMM models saved.')
fprintf('\n')

%% RANDOM FOREST RESULTS
[matRF,accRF,labels] = createConfusionMatrix(codesTrue_home,codesRF);
[matHmm,accHmm,labels] = createConfusionMatrix(codesTrue_home,codesHmm);
disp(['accRF = ' num2str(accRF)]);
disp(['accHMM = ' num2str(accHmm)]);

%% SUMMARY FIGURES
%Temporal
figure
if drawplot.actvstime
    dt = cData.clipdur * (1-cData.clipoverlap);
    t = 0:dt:dt*(length(codesTrue_home))-1;  
    subplot(211), hold on
    plot(t,codesTrue_home,'.-g')
    plot(t,codesRF+.1,'.-r')
    plot(t,codesHmm+.2,'.-b')    
    xlim([0 t(end)])
    xlabel('Time [s]')
    legend('True','RF','HMM')
    ylim([0.5 nstates+0.5]);
    set(gca,'YTick',cell2mat(StateCodes(:,2))')
    set(gca,'YTickLabel',StateCodes(:,1))
    
    subplot(212)
    plot(t,max(P_RF,[],2),'r'), hold on
    line([0 t(end)],[1/nstates 1/nstates])
    xlim([0 t(end)])    
end

%Confusion Matrix
if drawplot.confmat
    figure('name',['Confusion Matrix']); hold on
    correctones = sum(matRF,2);
    correctones = repmat(correctones,[1 size(StateCodes,1)]);
    subplot(121); imagesc(matRF./correctones); colorbar
    set(gca,'XTick',[1:d],'XTickLabel',StateCodes(:,1))
    set(gca,'YTick',[1:d],'YTickLabel',StateCodes(:,1))
    axis square
    subplot(122); imagesc(matHmm./correctones); colorbar
    set(gca,'XTick',[1:d],'XTickLabel',StateCodes(:,1))
    set(gca,'YTick',[1:d],'YTickLabel',StateCodes(:,1))
    axis square
end

%% True/False Positive Distributions
figure('name','True/False Positives')
count = 1;
TF_mat = cell(2,5);

for ii = 1:length(uniqStates)
    idx = find(codesRF == ii);
    P_RF_activity = P_RF(idx,ii);
    TP_ind = find(codesRF(idx) == codesTrue_home(idx)'); %find indices of true postive
    FP_ind = find(~(codesRF(idx) == codesTrue_home(idx)')); %find indices of false postive
    
    TF_mat{1,ii} = [TF_mat{1,ii}; P_RF_activity(TP_ind)]; 
    TF_mat{2,ii} = [TF_mat{2,ii}; P_RF_activity(FP_ind)];
    
    bins = 20;
    [N_TP, ~] = histcounts(P_RF_activity(TP_ind),bins);
    [N_FP, ~] = histcounts(P_RF_activity(FP_ind),bins);
    y_max = max([N_TP N_FP])*1.2;
    
    subplot(2,5,count)
    histogram(P_RF_activity(TP_ind),bins)
    title(['True Postive for ' uniqStates{ii}])
    xlim([0 1])
    ylim([0 y_max])
    
    subplot(2,5,count+5)
    histogram(P_RF_activity(FP_ind),bins)
    title(['False Postive for ' uniqStates{ii}])
    xlim([0 1])
    ylim([0 y_max])

    count = count + 1;
end