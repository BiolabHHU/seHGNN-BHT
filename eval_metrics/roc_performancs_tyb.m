function [ fpr_tyb, tpr_tyb, AUCroc ] = roc_performancs_tyb( labels, scores , plot_flag)
%  Matlab Code-Library for Feature Selection
%
%  Before using the Code-Library, please read the Release Agreement carefully.
%
%  Release Agreement:
%
%  - All technical papers, documents and reports which use the Code-Library will acknowledge the use of the library as follows: 
%    锟絋he research in this paper use the Feature Selection Code Library (FSLib)锟? and a citation to:
%  
%  Version 5.0 August 2017
%  Support: Giorgio Roffo
%  E-mail: giorgio.roffo@glasgow.ac.uk

%  If you use our toolbox please cite our supporting papers:
% 
%  BibTex
%  ------------------------------------------------------------------------
% @InProceedings{RoffoICCV17, 
% author={Giorgio Roffo and Simone Melzi and Umberto Castellani and Alessandro Vinciarelli}, 
% booktitle={2017 IEEE International Conference on Computer Vision (ICCV)}, 
% title={Infinite Latent Feature Selection: A Probabilistic Latent Graph-Based Ranking Approach}, 
% year={2017}, 
% month={Oct}}
%  ------------------------------------------------------------------------
% @Inbook{Roffo2017,
% author="Roffo, Giorgio
% and Melzi, Simone",
% editor="Appice, Annalisa
% and Ceci, Michelangelo
% and Loglisci, Corrado
% and Masciari, Elio
% and Ra{\'{s}}, Zbigniew W.",
% title="Ranking to Learn:",
% bookTitle="New Frontiers in Mining Complex Patterns: 5th International Workshop, NFMCP 2016, Held in Conjunction with ECML-PKDD 2016, Riva del Garda, Italy, September 19, 2016, Revised Selected Papers",
% year="2017",
% publisher="Springer International Publishing",
% address="Cham",
% pages="19--35",
% isbn="978-3-319-61461-8",
% doi="10.1007/978-3-319-61461-8_2",
% url="https://doi.org/10.1007/978-3-319-61461-8_2"
% }
%  ------------------------------------------------------------------------
% @InProceedings{RoffoICCV15, 
% author={G. Roffo and S. Melzi and M. Cristani}, 
% booktitle={2015 IEEE International Conference on Computer Vision (ICCV)}, 
% title={Infinite Feature Selection}, 
% year={2015}, 
% pages={4202-4210}, 
% doi={10.1109/ICCV.2015.478}, 
% month={Dec}}
%  ------------------------------------------------------------------------
% EXAMPLE
% Call the function 
% roc_performancs( [1 1 1 1 -1 -1 -1 -1]', [0.2 .8 .1 .3 -.1 -.7 0.01 -0.05]' , 1)


[Xfpr,Ytpr,~,AUCroc]  = perfcurve(double(labels), double(scores), 1,'TVals','all','xCrit', 'fpr', 'yCrit', 'tpr');
[Xpr,Ypr,~,AUCpr] = perfcurve(double(labels), double(scores), 1, 'TVals','all','xCrit', 'reca', 'yCrit', 'prec');
[acc,~,~,~] = perfcurve(double(labels), double(scores), 1,'xCrit', 'accu');

prec = Ypr; prec(isnan(prec))=1;
tpr = Ytpr; tpr(isnan(tpr))=0; % recall = true positive rate
fpr = Xfpr; % (1 - Specificity)
recall = tpr;

% Compute F-Measure
f1= 2*(prec.*tpr) ./ (prec+tpr);
[Max_F1,idx] = max(f1);
F1_Precision = prec(idx);
F1_Recall = tpr(idx);
F1_accuracy = acc(idx);

fpr = round(fpr*10000)/10000;   % 量化一下

if plot_flag
%     figure;
%     subplot(1,2,1)
%     plot([tpr], [ prec], '-b', 'linewidth',2); % add pseudo point to complete curve
%     xlabel('recall');
%     ylabel('precision');
%     grid on
%     title(['precision-recall ']);
    
%     subplot(1,2,2)
%     plot(fpr, tpr, '-r', 'linewidth',2); % add pseudo point to complete curve
%     xlabel('false positive rate');
%     ylabel('true positive rate');
%     % grid on
%     title(['ROC curve']);
    
    for jj = 1:length([fpr])-1
        plot(fpr(jj:jj+1),tpr(jj)*ones(1,2), '-r', 'linewidth',2);
        hold on
        if jj>1
           plot([fpr(jj),fpr(jj)],[tpr(jj-1),tpr(jj)],'-r', 'linewidth',2);
        end
    end
    
    for jj = length([fpr]):length([fpr])
        plot([fpr(jj),fpr(jj)],[tpr(jj-1),tpr(jj)],'-r', 'linewidth',2);
    end
    
    xlabel('false positive rate');
    ylabel('true positive rate');
    % grid on
    title(['ROC curve']);
    
    
    
    
end

AUCroc = AUCroc; % Area Under the ROC curve
Acc = 100*sum(labels == sign(scores))/length(scores); % Accuracy

fpr_tyb = zeros(10001,1);
tpr_tyb = ones(10001,1);

for i = 1:length(fpr)-1
    tpr_tyb(round(fpr(i)*10000+1):round(fpr(i+1)*10000))=tpr(i);    
end


% fpr*10000 
% tpr

end

