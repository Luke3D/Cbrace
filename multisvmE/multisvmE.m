function [trainresult,testresult] = multisvmE(TrainingSet,GroupTrain,TestSet,kernel,rbf_sigma)
%Models a given training set with a corresponding group vector and 
%classifies a given test set using an SVM classifier according to a 
%one vs. all relation. 
%
%This code was written by Cody Neuburger cneuburg@fau.edu
%Florida Atlantic University, Florida USA
%This code was adapted and cleaned from Anand Mishra's multisvm function
%found at http://www.mathworks.com/matlabcentral/fileexchange/33170-multi-class-support-vector-machine/
%
%multisvmE takes the following inputs:
%   TrainingSet: NxM array of N training observations with M measurements
%       each
%   GroupTrain: 1xN vector of training class indicators
%   TestSet: PxM array of P testing observations with M measurements each
%   kernel: (optional) SVM kernel to use
%       default: polynomial
%   method: (optional) optimization method to use
%       defautl: SMO
%
%multisvmE returns the following outputs:
%   trainresult: 1xN vector of training class predictions
%   testresult: 1xP vector of testing class predictions
%
%Modified by Eric Earley, 2013
%   RIC, Center for Bionic Medicine

if nargin < 4 || isempty(kernel)
    kernel = 'polynomial';
end

p = gcp('nocreate');
if isempty(p)
    parpool('local')
end

u=unique(GroupTrain);
numClasses=length(u);

loopflag = true();
Trset = TrainingSet;
GrTrain = GroupTrain;
while loopflag;
    try
        parfor k=1:numClasses
            disp(['Class ',num2str(k)]);
            %Vectorized statement that binarizes Group
            %where 1 is the current class and 0 is all other classes
            G1vAll=(GrTrain==u(k));
%             opts = svmsmoset('MaxIter',15000); %increase maximum iterations
            opts = svmsmoset('MaxIter',15000,'kktviolationlevel',.05); %increase maximum iterations
%             models(k) = svmtrain(Trset,G1vAll,'kernel_function',kernel,...
%                 'method','SMO','smo_opts',opts); %#ok<AGROW>
            models(k) = svmtrainE(Trset,G1vAll,'kernel_function',kernel,...
                'method','SMO','rbf_sigma',rbf_sigma,'smo_opts',opts); %#ok<AGROW>
        end
        loopflag = false();
    catch exception
        if strcmp(exception.identifier,'Bioinfo:seqMinOpt:NoConvergence')
            if length(GrTrain)==length(GroupTrain)
                disp('No convergence reached. Performing CNN.');
                [Trset,GrTrain] = CNN(TrainingSet,GroupTrain);
            else
                disp('CNN data still did not converge. Aborting SVM.');
                error('multisvmE:NoConvergence','SVM did not converge with reduced dataset.');
            end
        else
            rethrow(exception);
        end
    end
    
end

disp(' ');
disp('Training.');
% classify train cases
trainresult = zeros(numClasses,size(TrainingSet,1));
for i=1:numClasses
%     [~,trainresult(i,:)] = svmclassify(models(i),TrainingSet);
    [~,trainresult(i,:)] = svmclassifyE(models(i),TrainingSet);
end
trainresult = vec2ind(bsxfun(@eq,trainresult,min(trainresult)));

disp('Testing.');
%classify test cases
testresult = zeros(numClasses,size(TestSet,1));
for i=1:numClasses
%     [~,testresult(i,:)] = svmclassify(models(i),TestSet);
    [~,testresult(i,:)] = svmclassifyE(models(i),TestSet);
end
testresult = vec2ind(bsxfun(@eq,testresult,min(testresult)));

disp(' ');
