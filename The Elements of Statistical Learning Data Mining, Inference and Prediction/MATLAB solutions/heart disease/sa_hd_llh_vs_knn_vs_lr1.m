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

% Initialize fitting parameters
initial_theta = zeros(size(X, 10),1);

% Set regularization parameter lambda to 1
lambda = 1;

% Compute and display initial cost and gradient for regularized logistic
% regression
[cost, grad] = costFunctionReg(initial_theta, X, y, lambda);

fprintf('Cost at initial theta (zeros): %f\n', cost);
fprintf('Expected cost (approx): 0.693\n');
fprintf('Gradient at initial theta (zeros) - first five values only:\n');
fprintf(' %f \n', grad(1:5));
fprintf('Expected gradients (approx) - first five values only:\n');
fprintf(' 0.0085\n 0.0188\n 0.0001\n 0.0503\n 0.0115\n');

fprintf('\nProgram paused. Press enter to continue.\n');

% Compute and display cost and gradient
% with all-ones theta and lambda = 10
test_theta = ones(size(X,10),1);
[cost, grad] = costFunctionReg(test_theta, X, y, 10);

fprintf('\nCost at test theta (with lambda = 10): %f\n', cost);
fprintf('Expected cost (approx): 3.16\n');
fprintf('Gradient at test theta - first five values only:\n');
fprintf(' %f \n', grad(1:5));
fprintf('Expected gradients (approx) - first five values only:\n');
fprintf(' 0.3460\n 0.1614\n 0.1948\n 0.2269\n 0.0922\n');

fprintf('\nProgram paused. Press enter to continue.\n');

% Initialize fitting parameters
initial_theta = zeros(size(X, 2), 1);

% Set regularization parameter lambda to 1 (you should vary this)
lambda = 25;

% Set Options
options = optimset('GradObj', 'on', 'MaxIter', 400);

% Optimize
[theta, J, exit_flag] = ...
	fminunc(@(t)(costFunctionReg(t, X, y, lambda)), initial_theta, options);

%Transforme y - data to logical matrix
y_l = logical(y);

% Compute accuracy on our training set
Y_pred_lr = predict(theta, X);

fprintf('Train Accuracy: %f\n', mean(double(Y_pred_lr == y_l)) * 100);

%Classification confusion matrix for loglistic regression 

ConfusionMatrix_lr = confusionmat(y_l ,Y_pred_lr);
[m,order] = confusionmat(y_l ,Y_pred_lr);
figure(3)
cm_lr = confusionchart(m,order);
n_lr = sum(ConfusionMatrix_lr ,'all'); % number of instances
diag_lr = diag(ConfusionMatrix_lr);% number of correctly classified instances per class
accuracy_lr = sum(diag_lr) / n_lr ;

