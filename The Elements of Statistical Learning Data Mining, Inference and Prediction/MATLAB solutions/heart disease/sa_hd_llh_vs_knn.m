%Initialization
clear ; close all; clc

%Import raw data
sa_hd_data = readmatrix("sa_hd");
x = sa_hd_data(:, [1:10]);
y = sa_hd_data(:, [11]);
% Z function to standardize input values
X = zscore(x,0);
% Fit for loglikelihood regression 
llh = fitglm(X,y);
%Predict for loglikelihood regression  
y_pred_llh = predict(llh, X);
Y_pred_llh =  zeros(1,462);
y_pred_llh = transpose(y_pred_llh);
for n = [1:462]
    if y_pred_llh(n) <= 0.5
        Y_pred_llh(n) = 0;
    else
       Y_pred_llh(n) = 1;
    end
end
%Classification confusion matrix for loglikelihood regression 
Y_pred_llh_t = transpose(Y_pred_llh); 
ConfusionMatrix_llh = confusionmat(y ,Y_pred_llh_t);
[m,order] = confusionmat(y ,Y_pred_llh_t);
figure(1)
cm_llh = confusionchart(m,order);
n_llh = sum(ConfusionMatrix_llh ,'all'); % number of instances
diag_llh = diag(ConfusionMatrix_llh);% number of correctly classified instances per class
%row_sums_llh = sum(ConfusionMatrix_llh, 2);%number of instances per class
%col_sums_llh = sum(ConfusionMatrix_llh); %number of predictions per class
%p_llh = row_sums_lr/n_llh;%number of predictions per class
%q_llh = col_sums_lr/n_llh;% distribution of instances over the predicted classes
accuracy_llh = sum(diag_llh) / n_llh ;
% Compute error rate  for liear logistic regression 
%precision_llh = diag_llh ./ col_sums_llh; 
%recall_llh = diag_llh ./ row_sums_llh;
%f1_llh = 2 .* precision_llh .* recall_llh ./ (precision_llh + recall_llh); 
%error_rate_llh = [precision_llh, recall_llh, f1_llh]; 

%Fit for K-nearest neighbor classification
knn = fitcknn(X,y,'NumNeighbors',21,'Standardize',1);
%Predict for K-nearest neighbor classification
Y_pred_knn = predict(knn, X);
%Classification confusion matrix for K-nearest neighbor
ConfusionMatrix_knn = confusionmat(y ,Y_pred_knn);
figure(2)
[m,order] = confusionmat(y ,Y_pred_knn);
cm_knn = confusionchart(m,order);
n_knn = sum(ConfusionMatrix_knn ,'all'); % number of instances
diag_knn = diag(ConfusionMatrix_knn);% number of correctly classified instances per class
accuracy_knn = sum(diag_knn) / n_knn ;

%Fit for logistic regression


