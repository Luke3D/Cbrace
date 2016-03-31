function [accelerations, date, activities, activity_fraction] = getTestClips(accdata, varargin)
% extracts clips of activity samples from recorded accleration data

% options: secs, resample_secs, datetime_format, data_columns
% secs - length of the samples in seconds
% resample_secs - how often to resample
% datetime_columns - 1 if datetime stamp, 2 if date then time columns
% activity_columns - number of columns used to collect labelled activities
% remove_ends - number of samples to remove from the ends
% activity_fraction - how much of a sample needs to be one activity to have that label
% max_rate - the maximum sampling rate
% min_rate - the minimum sampling rate (no gaps of more than 1/min_rate)

% returns as cell arrays:
% ret_acc  - 4 row matrix with time in seconds, then x,y,z accelerations
% ret_date - cell of Java format date time stamp (date/time separated by a space
% ret_act  - activity_columns wide cell array

% file formats include
% time in sec, x, y, z, 1-2 date/time columns, 0-2 activity columns
% file may or may not be contiguous in time, and activities may or may
% not change throughout

options = struct('secs',10, 'resample_secs', 2, ...
    'datetime_columns', 1, 'activity_columns', 1, ...
    'remove_ends', 2, 'activity_fraction', 0.8, ...
    'max_rate', 40, 'min_rate', 2);
optionNames = fieldnames(options);
nArgs = length(varargin);
if round(nArgs/2)~=nArgs/2
    error('extract_features needs propertyName/propertyValue pairs')
end
for pair = reshape(varargin,2,[]) %# pair is {propName;propValue}
    inpName = lower(pair{1}); %# make case insensitive
    if any(strmatch(inpName,optionNames))
        options.(inpName) = pair{2};
    else
        error('%s is not a recognized parameter name',inpName)
    end
end
activity_fraction = [];
accelerations = {};   % the acceleration vector that gets returned
date = {};     % will be observations x number of y_vals
activities = cell(options.activity_columns,1);

datafiles = expandFilenames(accdata, '/acc_*.csv');

if isempty(datafiles)
    disp('No data files found');
    return;
end

% the code used to read in the data files based on the inputs
scan_code = '%f%f%f%f';
for i = 1:(options.datetime_columns + options.activity_columns)
    scan_code = [scan_code '%s'];
end

for file = 1:length(datafiles)
    filename = datafiles{file};
    
    % skip old activity file formats
    if strfind(filename,'_act.csv') > 0
        continue
    end
    endOfFileName = regexp(filename,'/','split');
    disp(['Loading clips: ', endOfFileName{2}]);
    
    % read the data into arrays
    rfid = fopen(filename, 'r');
    data = textscan(rfid, scan_code,'Delimiter', ',');
    fclose(rfid);
    num_columns = length(data);
    
    % make sure there is data in all the lines
    num_rows = length(data{1}); % to start out
    for i = 2:num_columns
        num_rows = min(num_rows, length(data{i}));
    end
    for i = 1:num_columns
        data{i} = data{i}(1:num_rows);
    end
    
    currentRow = 1; % current row index
    if isempty(data{i})
        disp('File empty. Skipping and continuing.');
        continue;
    else
        startTime = data{1}(currentRow);
    end
    
    storeCount = 1;
    while currentRow < num_rows
        
        %begin going down data
        currentRow = currentRow + 1;
        currentTime = data{1}(currentRow);
        elapsedTime = currentTime - startTime;
        
        for i = 1:options.activity_columns
            activityStore{storeCount,i} = data{end-options.activity_columns+i}(currentRow);
        end
        %store the activities and times
        if options.activity_columns == 2
            jointStore{storeCount,1} = ...
                [char(activityStore{storeCount,1}) '-' char(activityStore{storeCount,2})];
        else
            jointStore{storeCount,1} = char(activityStore{storeCount,1});
        end
        timeStore(storeCount) = data{1}(currentRow);
        storeCount = storeCount + 1;
        
        %condition where we need to save stuff
        if elapsedTime - options.secs > 0
            
            startInd = find(data{1} == startTime);
            endInd = currentRow;
            
            %get the activities in this clip
            uniqActivities = unique(jointStore);
            
            %go through an calculate Percent in each activity
            totalTimeSpent = zeros(length(uniqActivities),1);
            for i = startInd+1:endInd
                if options.activity_columns == 2
                    tempJointName = [char(data{end-options.activity_columns+1}(i)) '-'...
                        char(data{end-options.activity_columns+2}(i))];
                else
                    tempJointName = char(data{end-options.activity_columns+1}(i));
                end
                ind = find(strcmp(tempJointName,uniqActivities));
                totalTimeSpent(ind) = totalTimeSpent(ind) + data{1}(i) - data{1}(i-1);
            end
            
            percentTimeSpent = totalTimeSpent ./ elapsedTime;
            
            %find the
            acc = zeros(4,currentRow-startInd+1);
            for d = 1:4
                acc(d,:) = data{d}(startInd:currentRow);
            end
            accelerations{end+1} = acc;
            %reset the counter
            
            if options.datetime_columns == 1
                date{end+1} = data{5}{currentRow-1};
            elseif options.datetime_columns == 2
                date{end+1} = [data{5}{currentRow-1} ' ' data{6}{currentRow-1}];
            else
                disp('date columns options must be set to 1 or 2');
                return
            end
            [percentTimeSpent, sortedInd] = sort(percentTimeSpent,1,'descend');
            uniqActivities = uniqActivities(sortedInd);
            for k = 1:length(uniqActivities)
               tempStr{k} = regexp(uniqActivities{k},'-','split'); 
            end
            for act_i = 1:options.activity_columns
                activities{act_i}{end+1} = tempStr{1}{act_i};
            end
            activity_fraction(end+1) = percentTimeSpent(1);
            startTime = currentTime;
            activityStore = {};
            jointStore = {};
            timeStore = [];
            totalTimeSpent = [];
            percentTimeSpent = [];
            storeCount = 1;
        end
    end
end
end
