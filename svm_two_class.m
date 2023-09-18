function [train_h0_data, train_h0_label, train_h1_data, train_h1_label, test_h0_label, test_h1_label,testlabel]=svm_two_class(index)
% Include dependencies

addpath('./lib');                      % dependencies
addpath('./methods');            % FS methods
addpath(genpath('./lib/drtoolbox'));

listFS = {'ILFS','InfFS','ECFS','mrmr','relieff','mutinffs','fsv','laplacian','mcfs','rfe','L0','fisher','UDFS','llcfs','cfs'};
selection_method = listFS{10};           % Selected rfe

%numF = 54;
% Load the data and select features for classification
load(['ADS18.mat'])
load(['ADS18norm_cell.mat'])


Y(find(Y~=0))=1;
testlabel=Y(index);

[train_h0_data, train_h0_label, test_h0_label] = train_h0(index, X,X_cell, Y, selection_method, 0);

[train_h0_data, train_h0_label] = energy_normalization(train_h0_data, train_h0_label);

[train_h1_data, train_h1_label, test_h1_label] = train_h0(index, X, X_cell,Y, selection_method,1);

[train_h1_data, train_h1_label] = energy_normalization(train_h1_data, train_h1_label);

end

function [train_data_out, train_label_out] = energy_normalization(train_data, train_label)

    tmp = train_data';
    sample_energy_tmp = sqrt(sum(tmp.^2));
    
    agv_energy_1 = mean(sample_energy_tmp(train_label));
    avg_energy_0 = mean(sample_energy_tmp(~train_label));
    sizeoftmp = size(tmp);
    sample_energy_map = ones(1, sizeoftmp(2));
    sample_energy_map(train_label) = agv_energy_1;
    sample_energy_map(~train_label) = avg_energy_0;
    
    energy_map = ones(size(tmp,1),1) * sample_energy_map;
    
    train_data_out = (tmp ./ energy_map)'; 
    train_label_out = train_label;
    
    return
end

function [train_h0_data, train_h0_label,  test_h0_label] = train_h0(index, X, X_cell,Y, selection_method,tag)

test_h0_label = tag;
X_train = double(X);
Y(index)=tag;
Y_temp= nominal(ismember(Y,0)); 
Y_train = (double(Y_temp)-1)*2-1;   
% numF = 54;
%  labels:  neg_class: -1, pos_class: +1
    %selected FC number . Important parameter!

    switch lower(selection_method)
        case 'rfe'
            ranking = SVM_RFE_tyb_groupfast(Y_train, X_train, 10, 3);
    end

    %k = numF;     
     for i=[1:100]
          X_cell{1,i} = reshape(X_cell{1,i}(:,ranking(1:10))',[30,1]);
     end
    X = cell2mat(X_cell)';

    train_h0_data = X;
    train_h0_label = Y;

    train_h0_data(index,:)=[];
    train_h0_label(index)=[];

    [~,indx_1] = sort(train_h0_label,'descend');
    train_h0_label= logical(train_h0_label(indx_1));
    train_h0_data = train_h0_data(indx_1,:);

    return
end


