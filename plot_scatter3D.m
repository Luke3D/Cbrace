%close all

%Load any features .mat file
labels_activity = features_data.activity_labels;
nonwear_ind = [strmatch('Not Wearing',labels_activity,'exact'); ...
               strmatch('Stand to Sit',labels_activity,'exact'); ...
               strmatch('Sit to Stand',labels_activity,'exact'); ...
               strmatch('Misc',labels_activity,'exact'); ...
               strmatch('Lying',labels_activity,'exact')];
statesTrue = features_data.activity_labels;
codesTrue = zeros(1,length(statesTrue));
features = features_data.features;
sessionID = features_data.sessionID;

statesTrue(nonwear_ind) = [];
codesTrue(nonwear_ind) = [];
features(nonwear_ind,:) = [];
sessionID(nonwear_ind) = [];


uniqStates  = unique(statesTrue);

for i = 1:length(statesTrue)
    codesTrue(i) = find(strcmp(statesTrue{i},uniqStates));
end

s_idx = find(~(sessionID == 2));
codesTrue(s_idx) = [];
features(s_idx,:) = [];

%3D Scatter Plot
figure
f_idx = [8 43 53];
for ii = 1:length(uniqStates)
    hold on
    idx = find(codesTrue == ii);
    %idx = idx(1:10);
    S = repmat(10*10,length(idx),1);
    scatter3(features(idx,f_idx(1)),features(idx,f_idx(2)),features(idx,f_idx(3)),S,'filled')
end
grid on
%title('Separation of Activities in Feature Space','FontSize',18)
xlim([0 4])
zlim([0 50])
ylim([20 45])
% xlabel(features_data.feature_labels(f_idx(1)),'FontSize',16)
% ylabel(features_data.feature_labels(f_idx(2)),'FontSize',16)
% zlabel(features_data.feature_labels(f_idx(3)),'FontSize',16)
xlabel('X Std Dev','FontSize',24)
ylabel('Mean of Squares','FontSize',24)
zlabel('XY Cross Product','FontSize',24)
set(gca,'Box','off','TickDir','out','LineWidth',2,'FontSize',14,'FontWeight','bold','XGrid','on');
%h_legend = legend('Sitting','Stairs Dw','Stairs Up','Standing','Walking','Location','northeast');
%set(h_legend,'FontSize',20);
view(41,12)