%Import a csv file as t x y z variables
figure
hold on
plot(t,x,'r','LineWidth',4)
plot(t,y,'Color',[0 .5 0],'LineWidth',4)
plot(t,z,'b','LineWidth',4)
xlim([0.655*t(end) 0.795*t(end)])
ylim([-15 25])
xlabel('Time (s)','FontSize', 24)
ylabel('Acceleration (m/s^2)','FontSize', 24)
set(gca,'Box','off','XTick',[],'TickDir','out','LineWidth',2,'FontSize',24,'FontWeight','bold');
h_legend = legend('X','Y','Z','Location','Northwest','Orientation','Horizontal');
set(h_legend,'FontSize',20);
hold off