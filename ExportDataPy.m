%export data from healthy and patients
load trainData_healthy.mat

% HealthyData = table();
% % var1 = 'SubjID';
% var2 = 'Session';
% var3 = 'Features';
% var4 = 'Label';

SubjID = trainingClassifierData.subjectID;
Session = trainingClassifierData.sessionID;
Features = trainingClassifierData.features;

%convert activity labels to numeric codes
activities = unique(trainingClassifierData.activity);
StateCodes = cell(length(activities),2);
StateCodes(:,1) = activities;
StateCodes(:,2) = num2cell(0:length(activities)-1); %sorted by unique

for i = 1:length(activities)
    %recode activities with numbers
    inds = strcmp(trainingClassifierData.activity,activities(i));
    Label(inds) = i-1;
end
Label = Label';

Healthydata = table(SubjID,Session,Features,Label);

writetable(Healthydata,'./Export/HealthyData.csv')

%% Patient CBR data
clear all

load trainData_patient.mat %all the patients data (CBR + SCO)
% Isolate brace data (write lines here or use IsolateData.m)
brace = cellfun(@(x) x(7:9),trainingClassifierData.subject,'UniformOutput', false);
Cbrind = strcmp(brace,'Cbr');   %indices of Cbrace Data

%the relevant fields filtered by brace
CbrData.subjectID = trainingClassifierData.subjectID(Cbrind);
CbrData.subject = trainingClassifierData.subject(Cbrind);
CbrData.activity = trainingClassifierData.activity(Cbrind);
CbrData.features = trainingClassifierData.features(Cbrind,:);
CbrData.featureLabels = trainingClassifierData.featureLabels;
CbrData.sessionID = trainingClassifierData.sessionID(Cbrind);
trainingClassifierData = CbrData; %removed useless fields

%convert activity labels to numeric codes
activities = unique(trainingClassifierData.activity);
StateCodes = cell(length(activities),2);
StateCodes(:,1) = activities;
StateCodes(:,2) = num2cell(0:length(activities)-1); %sorted by unique

for i = 1:length(activities)
    %recode activities with numbers
    inds = strcmp(trainingClassifierData.activity,activities(i));
    Label(inds) = i-1;
end
Label = Label';

%create a table to export in csv format
SubjID = trainingClassifierData.subjectID;
Session = trainingClassifierData.sessionID;
Features = trainingClassifierData.features;

CBRdata = table(SubjID,Session,Features,Label);

writetable(CBRdata,'./Export/PatientCBRData.csv')


%% SCO data
clear all

load trainData_patient.mat %all the patients data (CBR + SCO)
% Isolate brace data (write lines here or use IsolateData.m)
brace = cellfun(@(x) x(7:9),trainingClassifierData.subject,'UniformOutput', false);
Cbrind = strcmp(brace,'SCO');   %indices of Cbrace Data

%the relevant fields filtered by brace
CbrData.subjectID = trainingClassifierData.subjectID(Cbrind);
CbrData.subject = trainingClassifierData.subject(Cbrind);
CbrData.activity = trainingClassifierData.activity(Cbrind);
CbrData.features = trainingClassifierData.features(Cbrind,:);
CbrData.featureLabels = trainingClassifierData.featureLabels;
CbrData.sessionID = trainingClassifierData.sessionID(Cbrind);
trainingClassifierData = CbrData; %removed useless fields

%convert activity labels to numeric codes
activities = unique(trainingClassifierData.activity);
StateCodes = cell(length(activities),2);
StateCodes(:,1) = activities;
StateCodes(:,2) = num2cell(0:length(activities)-1); %sorted by unique

for i = 1:length(activities)
    %recode activities with numbers
    inds = strcmp(trainingClassifierData.activity,activities(i));
    Label(inds) = i-1;
end
Label = Label';

SubjID = trainingClassifierData.subjectID;
Session = trainingClassifierData.sessionID;
Features = trainingClassifierData.features;

SCOdata = table(SubjID,Session,Features,Label);

writetable(SCOdata,'./Export/PatientSCOData.csv')
