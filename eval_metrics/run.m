clear;close;clc;

%a = xlsread('results\NI.xlsx');
%a = xlsread('results\PU.xlsx');
%a = xlsread('results\NYU.xlsx');
%a = xlsread('results\KKI.xlsx');
a = xlsread('results\PU_1.xlsx');

tpr_agv = zeros(10001,1);
AUC_agv = 0;
for i = 1:min(size(a,1),1000)
    nums = a(i,:);
    predict = [ones(1,nums(1)) zeros(1,nums(2)) zeros(1,nums(3)) ones(1,nums(4))];
    ground_truth = [ones(1,nums(1)) ones(1,nums(2)) zeros(1,nums(3)) zeros(1,nums(4))];
    % result=plot_roc(predict,ground_truth);
    
    
   % [fpr, tpr, AUC(i)] = roc_performancs_tyb( ground_truth', predict', 0);
   [fpr, tpr, AUC(i)] = roc_performancs_tyb( ground_truth, predict, 0);

    tpr_agv = tpr_agv + tpr;

%     axis([0 1 0 1]);
%     hold on
end
%save('Peking_1_plot.mat','tpr_agv');

AUC_agv = mean(AUC);
[AUC_min, AUC_min_indx]= min(AUC);
[AUC_max, AUC_max_indx]= max(AUC);
%tpr_agv = tpr_agv/min(size(a,1),50);
tpr_agv = tpr_agv/1000;

plot([0:0.0001:1], tpr_agv,'-r', 'linewidth',2);
hold on

% % AUC_min
% nums = a(AUC_min_indx,:);
% predict = [ones(1,nums(1)) zeros(1,nums(2)) zeros(1,nums(3)) ones(1,nums(4))];
% ground_truth = [ones(1,nums(1)) ones(1,nums(2)) zeros(1,nums(3)) zeros(1,nums(4))];
% 
% [fpr, tpr, AUC_min] = roc_performancs_tyb( ground_truth', predict', 0);
% plot([0:0.0001:1], tpr,'--b', 'linewidth',2);
% 
% % AUC_max
% nums = a(AUC_max_indx,:);
% predict = [ones(1,nums(1)) zeros(1,nums(2)) zeros(1,nums(3)) ones(1,nums(4))];
% ground_truth = [ones(1,nums(1)) ones(1,nums(2)) zeros(1,nums(3)) zeros(1,nums(4))];
% 
% [fpr, tpr, AUC_max] = roc_performancs_tyb( ground_truth', predict', 0);
% plot([0:0.0001:1], tpr,':b', 'linewidth',2);
% 
% if AUC_max == 1
%    plot([0 0], [0 1],':b', 'linewidth',2);
% end





aaa=1;

% nums = [117	1	97	1];
% predict = [ones(1,nums(1)) zeros(1,nums(2)) zeros(1,nums(3)) ones(1,nums(4))];
% ground_truth = [ones(1,nums(1)) ones(1,nums(2)) zeros(1,nums(3)) zeros(1,nums(4))];
% result=plot_roc(predict,ground_truth);
% axis([0 1 0 1]);
% disp(result);