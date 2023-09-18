clc
clear

%List =dir('G:\ADHD\esamble\results\*.txt');
%k =length(List);
%for j=1:k
 % A = importdata(['G:\ADHD\esamble\results\',List(j).name]);
  %savePath= List(j).name(1:end-4);
  %save(savePath,'A');
%end
NYU_data_limbicpre = importdata('G:\ADHD\esamble\results\NYU_data_limbicpre.txt');
save('NYU_data_limbicpre.mat','NYU_data_limbicpre');