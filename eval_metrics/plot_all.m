clc
clear


load('Peking_1_plot.mat','tpr_agv');
Peking_1_tpr = tpr_agv/1000;
load('Peking_plot.mat','tpr_agv');
Peking_tpr = tpr_agv/1000;
load('NI_plot.mat','tpr_agv');
NI_tpr = tpr_agv/1000;
load('KKI_plot.mat','tpr_agv');
KKI_tpr = tpr_agv/1000;
load('NYU_plot.mat','tpr_agv');
NYU_tpr = tpr_agv/1000;

% h1 = plot([0:0.0001:1], NYU_tpr,'-r', 'linewidth',2);
% hold on
% plot([0,0], [0 NYU_tpr(1)],'-r', 'linewidth',2);
% 
% h2 = plot([0:0.0001:1], Peking_tpr,'-b', 'linewidth',2);
% plot([0,0], [0 Peking_tpr(1)],'-b', 'linewidth',2);
% 
% h3 = plot([0:0.0001:1], KKI_tpr,'--m', 'linewidth',2);
% plot([0,0], [0 KKI_tpr(1)],'--m', 'linewidth',2);
% 
% h4 = plot([0:0.0001:1], Peking_1_tpr,':k', 'linewidth',2);
% plot([0,0], [0 Peking_1_tpr(1)],':k', 'linewidth',2);
% 
% h5 = plot([0:0.0001:1], NI_tpr,'-.c', 'linewidth',1.5);
% plot([0,0], [0 NI_tpr(1)],'-.c', 'linewidth',1.5);
% %axis([0 0.25 0.4 1])
% % axis([0 1 0 1])
% 
% 
% xlabel('False positive rate');
% ylabel('True positive rate');
% title(['ROC curve']);
%set(gca,'XTick',0:0.05:0.25);
%set(gca,'YTick',0.4:0.1:1);


% 
% figure
% plot([0:0.01:1],10*ones(101,1),'-r', 'linewidth',2);
% hold on
% plot([0:0.01:1],20*ones(101,1),'-b', 'linewidth',2);
% plot([0:0.01:1],30*ones(101,1),'--m', 'linewidth',2);
% plot([0:0.01:1],40*ones(101,1),':k', 'linewidth',2);
% plot([0:0.01:1],50*ones(101,1),'-.c', 'linewidth',2);
% axis([-1 2 0 60])



h1 = plot([0:0.0001:1], NYU_tpr,'-r', 'linewidth',1.2);
hold on
plot([0,0], [0 NYU_tpr(1)],'-r', 'linewidth',1.2);

h2 = plot([0:0.0001:1], Peking_tpr,'-b', 'linewidth',1.2);
plot([0,0], [0 Peking_tpr(1)],'-b', 'linewidth',1.2);

h3 = plot([0:0.0001:1], KKI_tpr,'--r', 'linewidth',1.2);
plot([0,0], [0 KKI_tpr(1)],'--r', 'linewidth',1.2);

h4 = plot([0:0.0001:1], Peking_1_tpr,':b', 'linewidth',1.2);
plot([0,0], [0 Peking_1_tpr(1)],':b', 'linewidth',1.2);

h5 = plot([0:0.0001:1], NI_tpr,'-.r', 'linewidth',1.2);
plot([0,0], [0 NI_tpr(1)],'-.r', 'linewidth',1.2);
%  axis([0 0.15 0.8 1])
%  axis([0 1 0 1])


xlabel('False positive rate');
ylabel('True positive rate');
% set(axes,'Position',[10 10 26 220]);
% %axes('position',[0.35,0.34,0.4,0.5]);     %关键在这句！�?画的小图
% axes('position',[0.38,0.32,0.4,0.5]);
% h1 = plot([0:0.0001:1], NYU_tpr,'-r', 'linewidth',1.2);
% NYU=sum(NYU_tpr)/10000;
% hold on
% plot([0,0], [0 NYU_tpr(1)],'-r', 'linewidth',1.2);
% 
% h2 = plot([0:0.0001:1], Peking_tpr,'-b', 'linewidth',1.2);
% plot([0,0], [0 Peking_tpr(1)],'-b', 'linewidth',1.2);
% PU=sum(Peking_tpr)/10000;
% h3 = plot([0:0.0001:1], KKI_tpr,'--r', 'linewidth',1.2);
% plot([0,0], [0 KKI_tpr(1)],'--r', 'linewidth',1.2);
% KKI=sum(KKI_tpr)/10000;
% h4 = plot([0:0.0001:1], Peking_1_tpr,':b', 'linewidth',1.2);
% plot([0,0], [0 Peking_1_tpr(1)],':b', 'linewidth',1.2);
% PU1=sum(Peking_1_tpr)/10000;
% h5 = plot([0:0.0001:1], NI_tpr,'-.r', 'linewidth',1.2);
% plot([0,0], [0 NI_tpr(1)],'-.r', 'linewidth',1.2);
% NI=sum(NI_tpr)/10000;
% axis([0 0.30 0.50 1])
% set(gca,'XTick',0:0.05:0.30);
% set(gca,'YTick',0.50:0.05:1);
legend([h1 h2 h3 h4 h5],{'NYU','PU','KKI','PU_1','NI'});


hold on













