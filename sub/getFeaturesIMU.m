function [fvec flab] = getFeaturesIMU(IMUdata)
% for running the SVM, we definitely should not need the abs stuff.

% this function performs interpolation and
% extracts features from the last secs in acc
%
% INPUTS
% acc:  a 7 x samples vector: 1 is time, 2-4 are xyz accelerations
%      pre is for prior to interpolation
% secs: the number of seconds to extract
% rate: the rate to interpolate


% features for each of the three channels for each sensor (acc and gyro)
fvec = [];
flab = {};

for sensor = 1:2
    if sensor == 1  %accelerometer
        S = IMUdata(2:4,:);
    else            %gyro
        S = IMUdata(5:7,:);
    end
    
    for i=1:size(S,1)
        % looking at separate timescales
        fvec = [fvec nanmean(S(i,:))]; flab = [flab; 'mean acc'];
        %     S5 = conv(S(i,:),gausswin(5)/nansum(gausswin(5)));
        %     S10 = conv(S(i,:),gausswin(10)/nansum(gausswin(10)));
        %     fvec = [fvec sqrt(nanmean(S5(:).^2))]; flab = [flab; '5 smooth rms'];
        %     fvec = [fvec sqrt(nanmean(S10(:).^2))]; flab = [flab; '10 smooth rms'];
        %     fvec = [fvec abs(nanmean(S(i,:)))]; flab = [flab; 'abs mean acc'];
        %     fvec = [fvec sqrt(nanmean(S(i,:).^2))]; flab = [flab; 'rms'];
        %     fvec = [fvec nanmax(S(i,:))];  flab = [flab; 'max'];
        %     fvec = [fvec nanmin(S(i,:))];  flab = [flab; 'min'];
        %     fvec = [fvec abs(nanmax(S(i,:)))];  flab = [flab; 'abs max'];
        %     fvec = [fvec abs(nanmin(S(i,:)))];  flab = [flab; 'abs min'];
        %
        % normalized histogram of the values
        histvec = histc((S(i,:)-nanmean(S(i,:))/nanstd(S(i,:))),[-3:1:3]);
        % remove the last data point (which is zero in almost all cases)
        %histvec = histvec(1:end-1); %Not required (Luca & Sohrab)
        fvec = [fvec histvec]; flab = [flab; cellstr(repmat('hist',7,1))];
        
        % 2nd, 3rd, 4th moments
        fvec = [fvec nanstd(S(i,:))];  flab = [flab; 'std acc '];
        
        %Fix NaN results when std deviation of clip is 0 (Actigraph data has this isssue)
        %add eps to stddev of data to avoid division by 0
        if nanstd(S(i,:)) == 0
            X = S(i,:); N = length(X);
            s = 1/N*sum((X-mean(X)).^3)/( sqrt(1/N*sum((X-mean(X)).^2)) + eps )^3; %skewness
            k = 1/N*sum((X-mean(X)).^4)/( 1/N*sum((X-mean(X)).^2) + eps )^2; %kurtosis
            fvec = [fvec s]; flab = [flab; 'skew'];
            fvec = [fvec k]; flab = [flab; 'kurt'];
        else
            fvec = [fvec skewness(S(i,:))]; flab = [flab; 'skew'];
            fvec = [fvec kurtosis(S(i,:))]; flab = [flab; 'kurt'];
        end
        
        % fourier transform
        %     N = 16;
        %     Y = fft(S(i,:),N);
        %     Pyy = Y.* conj(Y) / N;
        %     Pyy = Pyy(1:N/2); %Bug Fixed (Luca & Sohrab)
        %     Pyy = Pyy/nansum(Pyy.^2);
        %     fvec = [fvec Pyy];
        %     flab = [flab; cellstr(repmat('fourier',N/2,1))];
        %
        % moments of the difference
        fvec = [fvec sqrt(nanmean(diff(S(i,:)).^2))]; flab = [flab; 'mean diff'];
        fvec = [fvec nanstd(diff(S(i,:)))]; flab = [flab; 'std diff'];
        
        %Fix NaN results when std deviation of clip is 0 (Actigraph data has this isssue)
        %add eps to stddev of data to avoid division by 0
        if nanstd(diff(S(i,:))) == 0
            X = diff(S(i,:)); N = length(X);
            s = 1/N*sum((X-mean(X)).^3)/( sqrt(1/N*sum((X-mean(X)).^2)) + eps )^3; %skewness
            k = 1/N*sum((X-mean(X)).^4)/( 1/N*sum((X-mean(X)).^2) + eps )^2; %kurtosis
            fvec = [fvec s]; flab = [flab; 'skew diff'];
            fvec = [fvec k]; flab = [flab; 'kurt diff'];
        else
            fvec = [fvec skewness(diff(S(i,:)))]; flab = [flab; 'skew diff'];
            fvec = [fvec kurtosis(diff(S(i,:)))]; flab = [flab; 'kurt diff'];
        end
        %
        %     %max of jerk
        %     jerk = diff(S(i,:));
        %     fvec = [fvec max(abs(jerk))];
        %     flab = [flab; 'max jerk'];
        
    end
    
    % features that apply across the signals
    
    % overall mean
    fvec = [fvec nanmean(nanmean(S.^2))];
    flab = [flab; 'overall mean'];
    
    
    
    % for quasi angles
    S2=S/sqrt(nansum(S.^2));
    fvec = [fvec nanmean(S2(1,:).*S2(2,:))]; flab = [flab; 'cross prod 1'];
    fvec = [fvec nanmean(S2(1,:).*S2(3,:))]; flab = [flab; 'cross prod 2'];
    fvec = [fvec nanmean(S2(2,:).*S2(3,:))]; flab = [flab; 'cross prod 3'];
    fvec = [fvec abs(nanmean(S2(1,:).*S2(2,:)))]; flab = [flab; 'ABS cross prod 1'];
    fvec = [fvec abs(nanmean(S2(1,:).*S2(3,:)))]; flab = [flab; 'ABS cross prod 2'];
    fvec = [fvec abs(nanmean(S2(2,:).*S2(3,:)))]; flab = [flab; 'ABS cross prod 3'];
    
    % cross products
    fvec = [fvec nanmean(S(1,:).*S(2,:))]; flab = [flab; 'cross prod 1'];
    fvec = [fvec nanmean(S(1,:).*S(3,:))]; flab = [flab; 'cross prod 2'];
    fvec = [fvec nanmean(S(2,:).*S(3,:))]; flab = [flab; 'cross prod 3'];
    fvec = [fvec abs(nanmean(S(1,:).*S(2,:)))]; flab = [flab; 'abs cross prod 1'];
    fvec = [fvec abs(nanmean(S(1,:).*S(3,:)))]; flab = [flab; 'abs cross prod 2'];
    fvec = [fvec abs(nanmean(S(2,:).*S(3,:)))]; flab = [flab; 'abs cross prod 3'];
    
end
return


%------------------------------------------------------

function [w] = gausswin(M,alpha)
if nargin<2
    alpha=1;
end
n = -(M-1)/2 : (M-1)/2;
w = exp((-1/2) * (alpha * n/(M/2)) .^ 2)';

% % autocorrelation code
% laglim = 12;
% [Rx lags] = xcorr(sub_acc(1,:) - mean(sub_acc(1,:)), laglim, 'coeff');
% [Ry lags] = xcorr(sub_acc(2,:) - mean(sub_acc(2,:)), laglim, 'coeff');
% [Rz lags] = xcorr(sub_acc(3,:) - mean(sub_acc(3,:)), laglim, 'coeff');
% Rsum = (Rx + Ry + Rz) / 3;
% Rsum(1:10)'
