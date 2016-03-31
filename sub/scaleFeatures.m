function [data] = scaleFeatures(data)

fvec = data.features;
featureMax = max(fvec,[],1);
featureMin = min(fvec,[],1);
fvec = (fvec - repmat(featureMin,size(fvec,1),1))*spdiags(1./(featureMax-featureMin)',0,size(fvec,2),size(fvec,2));
data.features = fvec;

end

