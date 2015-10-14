function [D] = minkowski(X,Y,P,covar,Ndim)
% Calculate distance between points using Minkowski distance metric
%   [D] = minkowski(X,Y,P)
%   [D] = minkowski(...,covar,Ndim)
%
%minkowski takes the following inputs:
%   X: NxM observation points
%   Y: LxM target point
%   P: order of Minkowski distance
%       P = 1: Manhattan distance
%       P = 2: Euclidean distance
%   covar: (optional) (co)variance
%   	if empty, will calculate raw Minkowski distance
%       if vector, will calculate normalized Minkowski distance
%       if matrix, will calculate Mahalanobis Euclidean distance
%           (ignores P)
%	Ndim: (optional) number of dimensions to sum
%       if empty, will sum all dimensions
%           (M-dimension Manhattan distance)
%       if 1, Chebyshev distance
%       otherwise, n-dimension Manhattan distance
%
%minkowski returns the following outputs:
%   D: NxL vector of distances from X(n) to Y(l)
%
%Written by Eric Earley, 2014
%	RIC, Center for Bionic Medicine

[N,M] = size(X);
[L,M2] = size(Y);
assert(M == M2,'X and Y must match in dimension 2');

if nargin < 4
    covar = [];
end
if nargin < 5 || isempty(Ndim)
    Ndim = M;
    sortflag = false;
elseif Ndim == M
    sortflag = false;
else
    sortflag = true;
end

[C1,C2] = size(covar);
D = zeros(N,L);

for l=1:L
    if ~any([C1,C2]) %no covariance
        distances = abs(bsxfun(@minus,Y(l,:),X)).^P;
        if sortflag %prevent unnecessary sorting
            distances = sort(distances,2,'descend'); %sort each observation
            D(:,l) = ( sum(distances(:,1:Ndim),2) ).^(1/P);
        else
            D(:,l) = ( sum(distances,2) ).^(1/P);
        end
    elseif any([C1,C2]==1) %variance is vector
        if C2 == 1 %rotate vector if necessary
            covar = covar';
        end
        distances = bsxfun(@rdivide,abs(bsxfun(@minus,Y(l,:),X)).^P,covar);
        if sortflag
            distances = sort(distances,2,'descend'); %sort each observation
            D(:,l) = ( sum(distances(:,1:Ndim),2) ).^(1/P);
        else
            D(:,l) = ( sum(distances,2) ).^(1/P);
        end
    else %covariance matrix (Mahalanobis distance)
        %difference = bsxfun(@minus,Y(l,:),X);
        difference = bsxfun(@minus,X,Y(l,:));
        
        %this function takes advantage of MATLAB matrix multiplication speed
        %   and is used to prevent memory errors
        [dist] = matmult_byrow(difference,inv(covar));
        D(:,l) = sqrt(sum(dist.*difference,2));
        
        %    %old method
        %     for i=1:N
        %         %D(i) = sqrt(difference(i,:)*inv(covar)*difference(i,:)');
        %         D(i) = sqrt(difference(i,:)/covar*difference(i,:)');
        %     end
    end
end

end
