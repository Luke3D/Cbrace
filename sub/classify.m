function [fit, varargout] = classify(method, fvec, yvec, testvec, varargin)

%Revision by Luca 11/1/2013
% reads the feature vector in and returns the fit
% input:
% fvec - feature vector: observations x features
% yvec - the observed states
% testvec - the indices of the test data
% output:
% fit - The fitted model
%logistic regression variables to create the fit observations x 2 ??
% (No longer here) pred_states - the predicted states.  should be similar to yvec


%% INIT
if (nargin < 4)
    testvec = zeros(size(yvec));
end

if strcmp(method, 'svm')
    if length(varargin) < 1
        fit.c = 1;
    else
        fit.c = varargin{1};
    end
    if length(varargin) < 2
        fit.g = 0.1;
    else
        fit.g = varargin{2};
    end
elseif strcmp(method, 'smlr')
    if length(varargin) < 1
        fit.lambda = 0.0001;
    else
        fit.lambda = varargin{1};
    end
end
varargout{1} = [];
% check to see if the number of observations matches in the feature set as
% well as the observations

% nFeatures = size(fvec,2);
% nObservations = length(yvec);
if (length(yvec) ~= size(fvec,1))
    disp('number of observations doesn''t match in the input');
    %fit = NaN; pred_states = NaN; uStates=NaN;
    return;
end

[yvec_nums uStates] = cell2vec(yvec);

fit.uStates = uStates;
fit.min_fvec = min(fvec,[],1);
fit.max_fvec = max(fvec,[],1);
fit.removed_features = ~isnan(var(fvec));

% -----------------

% % scale the feature vector
% fvec = (fvec - repmat(fit.min_fvec,size(fvec,1),1))*spdiags(1./(fit.max_fvec-fit.min_fvec)',0,size(fvec,2),size(fvec,2));

% remove nans
fvec = fvec(:,fit.removed_features);
%% SVM
if strcmp(method,'svm')
    % TRAIN SVM
    % -q option for quiet mode
    % -b option for probability estimates
    % -c,-g parameters C and gamma
    
    fit.fit = svmtrain(yvec_nums(~testvec)', fvec(~testvec,:), ['-c ' num2str(fit.c) ' -g ' num2str(fit.g) ' -q -b 1']);
    
    %   %PREDICT (TEST) SVM
    %   model: SVM model structure from svmtrain.
    %   libsvm_options:
    %   -b probability_estimates: whether to predict probability estimates, 0 or 1 (default 0); one-class SVM not supported yet
    %   -q : quiet mode (no outputs)
    %   Returns:
    %   predicted_label: SVM prediction output vector.
    %   accuracy: a vector with accuracy, mean squared error, squared correlation coefficient.
    %   prob_estimates: If selected, probability estimate vector.
    [pred_num, acc, P] = svmpredict(yvec_nums(testvec)', fvec(testvec,:), fit.fit, ['-b 1']);
    varargout{2} = P;   %Probability estimate for each class
    varargout{3} = acc; %accuracy
    varargout{4} = pred_num; %predicted classes

    %% OTHER METHODS
elseif strcmp(method,'smlr')
    Y = zeros([length(yvec_nums) length(fit.uStates)]);
    for i = 1:length(yvec_nums)
        Y(i,yvec_nums(i)) = 1;
    end
    
    fit.fit = smlr(fvec(~testvec,:), Y(~testvec,:), 'lambda', fit.lambda, ...
        'max_iter', 1e5, 'fit_all', false, ...
        'mex', false, 'constant', true);
    
    % now determine the most likely classification
    
    % add a column of ones for the constant
    fvec = [ones(size(fvec,1), 1)  fvec];
    
    % Xw = X*w;   an N x M (observations by # features) matrix
    Xw = fvec * fit.fit;
    
    E = exp(Xw);
    S = sum(E,2) + 1; % or 1/smm
    P = E ./ (1 + S * ones(1,size(E,2)));
    P = [P 1-sum(P,2)];
    
    [dummy pred_num] = max(P,[],2);
    varargout{2} = P;
elseif strcmp(method,'naivebayes')
    nb = NaiveBayes.fit(fvec(~testvec,:),yvec_nums(~testvec));
    pred_num = predict(nb,fvec);
elseif strcmp(method,'knn')
    k = 3;
    pred_num = yvec_nums; % this will only be used for the training data
    pred_num(testvec) = knnclassify(fvec(testvec,:),fvec(~testvec,:),yvec_nums(~testvec),k);
elseif strcmp(method,'decisiontree')
    t = classregtree(fvec(~testvec,:),yvec_nums(~testvec),'method','classification');
    pred_num = str2num(cell2mat(eval(t,fvec)));
else
    disp(['That method is not available']);
    return
end
%%
% -----------------
varargout{1} = uStates(pred_num);
