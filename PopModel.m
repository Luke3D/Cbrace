% population model - Train a RF on each subject and combine the predictions
% for a new subject

%% Train a model on each subject and use majority voting to predict
% activities of remaining subject

%we first train on healthy and test on patients
load trainData_healthy
Nsubj = length(unique(trainingClassifierData.subject));
subjcodes = unique(trainingClassifierData.subjectID);

%coding activities with num and proportion of activities
activities = unique(trainingClassifierData.activity);
StateCodes = cell(length(activities),2);
StateCodes(:,1) = activities;
StateCodes(:,2) = num2cell(1:length(activities)); %sorted by unique

for i = 1:length(activities)
    actprop(i) = sum(double(strcmp(trainingClassifierData.activity,activities(i))))/length(trainingClassifierData.activity);
    %recode activities with numbers
    inds = strcmp(trainingClassifierData.activity,activities(i));
    trainingClassifierData.labels(inds) = i;
end
figure
bar(actprop)
set(gca,'XTick',1:length(activities),'XTickLabel',activities)

RFmodel = {};
parfor s = 1:Nsubj
        
    indtrain = trainingClassifierData.subjectID == subjcodes(s);
    Xtr = trainingClassifierData.features(indtrain,:);
    Ytr = trainingClassifierData.labels(indtrain);
    
    ntrees = 50;
    opts = statset('UseParallel',1);
    disp(['training model ', num2str(s)])
    RF = TreeBagger(ntrees,Xtr,Ytr,'Options',opts);
    RFmodel{s} = RF;
  
end


%% predict on each patient and average accuracies

%% split by brace type (use only C-brace data) for patients
patientdata = load('trainData_patient.mat');
patientdata = patientdata.trainingClassifierData;
brace = cellfun(@(x) x(7:9),patientdata.subject,'UniformOutput', false);
patientdata.brace = brace;
Cbrind = strcmp(patientdata.brace,'Cbr');   %indices of Cbrace Data
CbrData.subjectID = patientdata.subjectID(Cbrind);
CbrData.activity = patientdata.activity(Cbrind);
CbrData.features = patientdata.features(Cbrind,:);
CbrData.featureLabels = patientdata.featureLabels;
CbrData.sessionID = patientdata.sessionID(Cbrind);
%recode activities with numbers
for i = 1:length(activities) 
    inds = strcmp(CbrData.activity,activities(i));
    CbrData.labels(inds) = i;
end
subjcodes = unique(CbrData.subjectID);

%extract data from 1 patient
indte = CbrData.subjectID == 1;
Xte = CbrData.features(indte,:);
Yte = CbrData.labels(indte);

%predict with each model
for F = 1:length(RFmodel)
    disp(['predict from model ', num2str(F)]);
    [Yfit,scores(:,:,F)] = predict(RFmodel{F},Xte);
    Yfit = str2num(cell2mat(Yfit));
    YfitAll(:,F) = Yfit;    %the class predicted by each model
    PAll(:,F) = max(scores(:,:,F),[],2);
end

%combine the predictions by majority voting
Ypop = mode(YfitAll,2);

%compute accuracy, precision and recall
cmat_Pop = confusionmat(Yte,Ypop)
accPop = trace(cmat_Pop)/sum(sum(cmat_Pop))


%% now train a model on all 
ntrees = 50;
opts = statset('UseParallel',1);
Xtr = trainingClassifierData.features;
Ytr = trainingClassifierData.labels;
GlobalRF = TreeBagger(ntrees,Xtr,Ytr,'Options',opts);

[Yfit_Global,scores] = predict(GlobalRF,Xte);
Yfit_Global = str2num(cell2mat(Yfit_Global))
cmat_Global = confusionmat(Yte,Yfit_Global)
accGlobal = trace(cmat_Global)/sum(sum(cmat_Global))

figure
subplot(121), imagesc(cmat_Pop), axis square
subplot(122), imagesc(cmat_Global), axis square




