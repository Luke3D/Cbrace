function K = poly_kernelE(u,v,N,varargin)
%POLY_KERNEL polynomial kernel for SVM functions

% Copyright 2004-2008 The MathWorks, Inc.

if nargin < 3 || isempty(N)
    N = 3;
end

try
    dotproduct = (u*v');
catch %#ok<CTCH>
    dotproduct = sum(u.*v,2);
end

K = (dotproduct+1).^N;

%for i = 2:N
%    K = K.*(1 + dotproduct);
%end
