function [fvec, flab] = getFeaturesPAR(acc)
%Extract features from an accleration clip
%Used in the parallel code
% INPUTS
%acc: 3acc x time acceleration clip


% features for each of the three channels
fvec = [];
flab = {};
for i=1:size(acc,1)
    % looking at separate timescales
    fvec = [fvec nanmean(acc(i,:))]; flab = [flab; 'mean acc'];
%     S5 = conv(acc(i,:),gausswin(5)/nansum(gausswin(5)));
%     S10 = conv(acc(i,:),gausswin(10)/nansum(gausswin(10)));
%     fvec = [fvec sqrt(nanmean(S5(:).^2))]; flab = [flab; '5 smooth rms'];
%     fvec = [fvec sqrt(nanmean(S10(:).^2))]; flab = [flab; '10 smooth rms'];     
%     fvec = [fvec abs(nanmean(acc(i,:)))]; flab = [flab; 'abs mean acc'];
%     fvec = [fvec sqrt(nanmean(acc(i,:).^2))]; flab = [flab; 'rms'];
%     fvec = [fvec nanmax(acc(i,:))];  flab = [flab; 'max'];
%     fvec = [fvec nanmin(acc(i,:))];  flab = [flab; 'min'];
%     fvec = [fvec abs(nanmax(acc(i,:)))];  flab = [flab; 'abs max'];
%     fvec = [fvec abs(nanmin(acc(i,:)))];  flab = [flab; 'abs min'];
%     
    % normalized histogram of the values
    histvec = histc((acc(i,:)-nanmean(acc(i,:))/nanstd(acc(i,:))),[-3:1:3]);
    % remove the last data point (which is zero in almost all cases)
    %histvec = histvec(1:end-1); %Not required (Luca & Sohrab)
    fvec = [fvec histvec]; flab = [flab; cellstr(repmat('hist',7,1))];
    
    % 2nd, 3rd, 4th moments
    fvec = [fvec nanstd(acc(i,:))];  flab = [flab; 'std acc '];
    
    %Fix NaN results when std deviation of clip is 0 (Actigraph data has this isssue)
    %add eps to stddev of data to avoid division by 0
    if nanstd(acc(i,:)) == 0
        X = acc(i,:); N = length(X);
        s = 1/N*sum((X-mean(X)).^3)/( sqrt(1/N*sum((X-mean(X)).^2)) + eps )^3; %skewness
        k = 1/N*sum((X-mean(X)).^4)/( 1/N*sum((X-mean(X)).^2) + eps )^2; %kurtosis
        fvec = [fvec s]; flab = [flab; 'skew'];
        fvec = [fvec k]; flab = [flab; 'kurt'];
    else
        fvec = [fvec skewness(acc(i,:))]; flab = [flab; 'skew'];
        fvec = [fvec kurtosis(acc(i,:))]; flab = [flab; 'kurt'];
    end
    
    % fourier transform
%     N = 16;
%     Y = fft(acc(i,:),N);
%     Pyy = Y.* conj(Y) / N;
%     Pyy = Pyy(1:N/2); %Bug Fixed (Luca & Sohrab)
%     Pyy = Pyy/nansum(Pyy.^2);
%     fvec = [fvec Pyy];
%     flab = [flab; cellstr(repmat('fourier',N/2,1))];
%     
    % moments of the difference
    fvec = [fvec sqrt(nanmean(diff(acc(i,:)).^2))]; flab = [flab; 'mean diff'];
    fvec = [fvec nanstd(diff(acc(i,:)))]; flab = [flab; 'std diff'];
    
    %Fix NaN results when std deviation of clip is 0 (Actigraph data has this isssue)
    %add eps to stddev of data to avoid division by 0
    if nanstd(diff(acc(i,:))) == 0
        X = diff(acc(i,:)); N = length(X);
        s = 1/N*sum((X-mean(X)).^3)/( sqrt(1/N*sum((X-mean(X)).^2)) + eps )^3; %skewness
        k = 1/N*sum((X-mean(X)).^4)/( 1/N*sum((X-mean(X)).^2) + eps )^2; %kurtosis 
        fvec = [fvec s]; flab = [flab; 'skew diff'];
        fvec = [fvec k]; flab = [flab; 'kurt diff'];
    else
        fvec = [fvec skewness(diff(acc(i,:)))]; flab = [flab; 'skew diff'];
        fvec = [fvec kurtosis(diff(acc(i,:)))]; flab = [flab; 'kurt diff'];
    end
%     
%     %max of jerk
%     jerk = diff(acc(i,:));
%     fvec = [fvec max(abs(jerk))];
%     flab = [flab; 'max jerk'];
    
end

% features that apply across the signals

% overall mean
fvec = [fvec nanmean(nanmean(acc.^2))];
flab = [flab; 'overall mean'];



% for quasi angles
S2=acc/sqrt(nansum(acc.^2));
fvec = [fvec nanmean(S2(1,:).*S2(2,:))]; flab = [flab; 'cross prod 1'];
fvec = [fvec nanmean(S2(1,:).*S2(3,:))]; flab = [flab; 'cross prod 2'];
fvec = [fvec nanmean(S2(2,:).*S2(3,:))]; flab = [flab; 'cross prod 3'];
fvec = [fvec abs(nanmean(S2(1,:).*S2(2,:)))]; flab = [flab; 'ABS cross prod 1']; 
fvec = [fvec abs(nanmean(S2(1,:).*S2(3,:)))]; flab = [flab; 'ABS cross prod 2'];
fvec = [fvec abs(nanmean(S2(2,:).*S2(3,:)))]; flab = [flab; 'ABS cross prod 3'];

% cross products
fvec = [fvec nanmean(acc(1,:).*acc(2,:))]; flab = [flab; 'cross prod 1'];
fvec = [fvec nanmean(acc(1,:).*acc(3,:))]; flab = [flab; 'cross prod 2'];
fvec = [fvec nanmean(acc(2,:).*acc(3,:))]; flab = [flab; 'cross prod 3'];
fvec = [fvec abs(nanmean(acc(1,:).*acc(2,:)))]; flab = [flab; 'abs cross prod 1'];
fvec = [fvec abs(nanmean(acc(1,:).*acc(3,:)))]; flab = [flab; 'abs cross prod 2'];
fvec = [fvec abs(nanmean(acc(2,:).*acc(3,:)))]; flab = [flab; 'abs cross prod 3'];

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
