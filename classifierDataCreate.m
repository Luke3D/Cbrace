%AGGREGATE FEATURES/CLIPS ACROSS ALL THE SUBJECTS AND SAVES DATA TO BE
%LOADED BY THE MAIN.m FILE
%Modified by Luca 10/28/2013
function classifierDataCreate

addpath([pwd '/sub'])

proceed = 1;
while proceed > 0
    population = input('Are you analyzing healthy or patient or home? ','s');
    if strcmpi(population,'patient')
        population = 'patient';
        trainingFeatureDir = '../features_patient/';
        proceed = 0;
    elseif strcmpi(population,'healthy')
        population = 'healthy';
        trainingFeatureDir = '../features_healthy/';
        proceed = 0;
    elseif strcmpi(population,'home')
        population = 'home';
        trainingFeatureDir = '../features_home/';
        proceed = 0;
    else
        disp('Please type healthy or patient or home.');
        proceed = 1;
    end
end

classifierData = combineAllSubjectFeatures(trainingFeatureDir);
classifierData.states = createStateList(classifierData);
classifierData = removeDataWithNaNs(classifierData);

%ACTIVITIES TO REMOVE
classifierData = removeDataWithLocation(classifierData,'Not Wearing');
classifierData = removeDataWithActivity(classifierData,'Not Wearing'); 
classifierData = removeDataWithActivity(classifierData,'Misc');
classifierData = removeDataWithLocation(classifierData,'Not Wearing');
classifierData = removeDataWithActivity(classifierData,'Sit to Stand');
classifierData = removeDataWithActivity(classifierData,'Stand to Sit');
classifierData = removeDataWithActivity(classifierData,'Lying');
% classifierData = removeDataWithActivity(classifierData,'Stairs Up');
% classifierData = removeDataWithActivity(classifierData,'Stairs Dw');
% classifierData = removeDataWithActivity(classifierData,'Wheeling');

%NOT REQUIRED WHEN USING ONE LOCATION 
% classifierData = removeDataWithLocation(classifierData,'Hand (arm at side)');
% classifierData = combineLocations(classifierData,'Pocket');
% classifierData = combineLocations(classifierData,'Belt');
% classifierData = combineLocations(classifierData,'Bag');
% classifierData = combineLocations(classifierData,'Hand');
%fix annotation error: hand was accidentally labeled as belt
% classifierData = removeDataWithState(classifierData,'Belt/Misc');

trainingClassifierData = classifierData;

%add record time [s]
overlap = trainingClassifierData.clipoverlap;
Nclips = size(trainingClassifierData.features,1);
clipdur = trainingClassifierData.clipdur;
trainingClassifierData.recordtime = (round((1-overlap)*Nclips)+1)*clipdur;

%Add an ID for each of the clips (consecutive numbering)
clipID = [1:Nclips]';
% subjectID = zeros(Nclips,1);
% subjectID(:) = features_data.subjectID;

filename = ['trainData_' population];
save(filename,'trainingClassifierData','clipID');