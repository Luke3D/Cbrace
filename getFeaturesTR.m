function features = getFeaturesTR(posteriors)
    features = posteriors;
       
    %entropy = zeros(size(posteriors,1),1);
    variance = zeros(size(posteriors,1),1);
    %skew = zeros(size(posteriors,1),1);
    for z = 1:size(posteriors,1)
        %entropy(z) = -sum(posteriors(z,:).*log2(posteriors(z,:)));
        variance(z) = var(posteriors(z,:));
        %skew(z) = skewness(posteriors(z,:));
    end
    %features = [features entropy];
    features = [features variance];    
    %features = [features skew];
end