function r_out = SVM_RFE_tyb_fast(label, data, num_group, num_in_group)

% 当data为二维数组时
% num_in_group: 组内有几个特征 size（data，2）
% num_group：组数  size（data，1）
% r_out：组数的重要程度输出

    % SVM-RFE
    % SVM Recursive Feature Elimination (SVM RFE)
    % by liyang @BNU Math
    % Email:patrick.lee@foxmail.com
    % last modified 2010.09.18
    
%     data = randn(216,720);
%     label = randn(216,1)>0;
%     numclass=size(unique(label),1);
    numclass = 2;
    Num = size(data,2);
    
    
    s = [1:Num];
    s_tmp = s;
    r = [];
    r_tmp = r;
    iter = 1;  %  删除的特征个数

    if numclass==2

        
        % grouped SVM
        while ~isempty(s)
            X = data(:,s);
            % model = libsvmtrain(label, X);
            tmp =  size(s,2)/num_in_group - num_group;
        
            if tmp>0    % 继续排序           
               model = fitcsvm(X,label);   %by tyb
               w = model.SupportVectors' * (model.SupportVectorLabels.*model.Alpha);%%求权重  nSV 是各个类的支持向量数量   在sv_coef中每一行存放该支持向量与其它各类（n-1类）进行一对一训练时系数
               c = w.^2;
               if num_in_group >1
                 c_r = reshape(c,length(c)/num_in_group,num_in_group);
                 c_r_1 = sum(c_r');
               else
                   c_r = c;
                   c_r_1 = c;
               end
%                %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%                [~ , f_1] = min(c_r_1);
%                f = f_1:size(c_r,1):length(c);
%                r = [s(f),r];
%                s(f) = [];    % 里面是绝对索引，妙！
%                %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
               
               %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
               % 折半去除
               [~, c_indx] = sort(c_r_1,'ascend');
               num_reduce = floor(max(tmp,2)/2);  % 需要减少的组数
%                num_reduce = 1;
               tmp_c = c_indx(1:num_reduce);  % 选出组数的索引,相对索引
               if num_in_group==1
%                    tmp_c = tmp_c';
               else
                   tmp_c = tmp_c'*ones(1,num_in_group) + ones(length(tmp_c),1)*[0:num_in_group-1]*size(c_r,1);
               end
               tmp_c = tmp_c';
               r = [s(tmp_c(:)),r];
               s(tmp_c(:)) = [];    % 里面是绝对索引，妙！
               %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
               
            else
               % 满足要求后，直接填充r 
               s_r = reshape(s,length(s)/num_in_group,num_in_group);
               s_1 = s_r';          % 按每隔8个为一组
               r = [s_1(:)' r];      
               break;  
            end
        end

         r_out = r(1:num_in_group:end);   % r是8个一组！
                

        end
%         display(iter);
    end
% end