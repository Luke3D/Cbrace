%run this to generate the patient dataset used in the project

load trainData_patient
cData = isolateSession(trainingClassifierData,4,1); %training

for zz = 1:length(cData.subject)
    temp = char(cData.subject(zz));
    cData.subjectBrace(zz) = {temp(7:9)};
end
cData = isolateBrace(cData,'Cbr');  %Cbr or SCO data
%% Remove stairs from patients above
patient_stairs = [8 11 12 14 15 19];
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

save trainDataCbr2.mat cData