function datafiles = expandFilenames(accdata, wildcard)
  % this function expands the filename, directory names, or list of files and dirs
  % to a cell array of filenames

  % wildcard default should be '/acc_*.csv'
  % get all the filenames from the accdata list
  datafiles = [];
  
  if ~iscell(accdata)    % if it's not a cell array, make it one
      if isdir(accdata)
          accdata = {accdata};
      else % it must be a filename
          datafiles = {accdata};
          return;
      end
  end
  
  for df_i = 1:length(accdata)
    df = accdata{df_i};
    if isdir(df)
      dfiles = dir([df wildcard]);
      for i = 1:length(dfiles)
        if ~strcmp(dfiles(i).name(1),'.')
          % it's not '.' or '..'
          datafiles{end+1} = strcat(df,'/',dfiles(i).name);
        end
      end
    else % it is not a directory
      if exist(df,'file')
        datafiles{end+1} = df;  % assume it's a file then
      end
    end
  end