function [data] = scaleFeaturesAG(data)

fvec = data.features;
featureMax = max(fvec,[],1);
featureMin = min(fvec,[],1);

sz = size(fvec);
n_col = sz(2);

for ii = 1:n_col
    if (featureMax(ii)-featureMin(ii) == 0)
        fvec(:,ii) = 0;
    else
        fvec(:,ii) = (fvec(:,ii)-featureMin(ii))./(featureMax(ii)-featureMin(ii));
    end
end

data.features = fvec;

end

