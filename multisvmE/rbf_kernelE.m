function kval = rbf_kernelE(u,v,sigma,varargin)
%RBF_KERNEL radial basis function kernel for SVM functions

% Copyright 2004-2008 The MathWorks, Inc.
if nargin < 3
    sigma = 1;
else
   if ~isscalar(sigma)
        error('Bioinfo:rbfkernel:SigmaNotScalar',...
            'Sigma must be a scalar.');
    end
    if sigma == 0
        error('Bioinfo:rbfkernel:SigmaZero',...
            'Sigma must be non-zero.');
    end
end

kval = exp( (-1/(2*sigma^2)) * minkowski(u,v,2) );

% kval2 = exp(-(1/(2*sigma^2))*(repmat(sqrt(sum(u.^2,2).^2),1,size(v,1))...
%     -2*(u*v')+repmat(sqrt(sum(v.^2,2)'.^2),size(u,1),1)));
% 
% N1 = size(u,1);
% N2 = size(v,1);
% K = u*v'/sigma^2;
% d = diag(K);
% K = K-ones(N1,1)*d'/2;
% K = K-d*ones(1,N2)/2;
% kval3 = exp(K);