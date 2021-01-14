%Initialization
clear ; close all; clc

% Raw data
spamdata = readmatrix("spambase");
X = spamdata(:,1:57);
y = spamdata(:,58);

% Generate regressions 
% Ridge regression coefficient
k = 57;
B_ridge = ridge(y,X,k); 
y_pred_ridge = logsig(X*B_ridge)>=0.5;
Accuracy_ridge = mean(double(y_pred_ridge == y)) * 100;

% Lasso regression coefficient
B_lasso = lasso(X,y,'Alpha',0.75,'CV',10); 
B_lasso_T = transpose(B_lasso);
B_lasso_mean = mean(B_lasso_T);
B_lasso_mean_T = transpose(B_lasso_mean);
y_pred_lasso = logsig(X*B_lasso_mean_T)>=0.5;
Accuracy_lasso = mean(double(y_pred_lasso == y)) * 100;

% OLS regression coefficient
[b,se_b,mse,S] = lscov(X,y); 
B_ols = se_b;
y_pred_ols = logsig(X*B_ols)>=0.5;
Accuracy_ols = mean(double(y_pred_ols == y)) * 100;

% PLS regression coefficient
Y = y;
[XL,YL,XS,YS,BETA,PCTVAR] = plsregress(X,Y,57) ;
B_pls = BETA(1:57);
y_pred_pls = logsig(X*B_pls)>=0.5;
Accuracy_pls = mean(double(y_pred_pls == y)) * 100;

% PCR regression coefficient
[PCALoadings,PCAScores,PCAVar] = pca(X,'Economy',false);
B_pcr = regress(y-mean(y), PCAScores(:,1:57));
y_pred_pcr = logsig(X*B_pcr)>=0.5;
Accuracy_pcr = mean(double(y_pred_pcr == y)) * 100;

% Loglikelihood regression 
llh = fitglm(X,y);
y_pred_llh = predict(llh, X)>=0.5;
Accuracy_llh = mean(double(y_pred_llh == y)) * 100;
B = [0.200278623626125;-0.0498190288944038;-0.0120462356832266;0.0392785790665272;0.0119172871668733;0.0842077237694428;0.118840539691056;0.212940544302344;0.0939888974333136;0.0724740306621384;0.0150673955680904;0.0568567170128131;-0.0278594942276625;0.0119040396077204;0.00485997980117172;0.0185245238632741;0.0750613606291245;0.0517157134801211;0.0553979180302770;0.0141333699121773;0.0617219768990697;0.0526937283591076;0.0447669999733873;0.174800431470864;0.0908907931300598;-0.0231749677785555;-0.0216289521262606;-0.0122016023637028;0.00398741601190762;-0.00744954418357027;-0.0519483778814511;-0.0232939752859470;0.00633175996415026;-0.0419834167119737;0.0511417650913882;-0.0311687177117950;0.0264800624687861;-0.0332125168206092;-0.0534386057896991;-0.0197545953365619;0.0407608890126584;-0.00836408577103496;-0.0369265208814125;-0.0632389776030241;-0.0323800300873051;-0.0352532827577345;-0.0378141110806161;-0.195177399719784;-0.0582229385815656;-0.140102484185709;-0.0599593550410559;-0.0590515789957219;0.0680529871324928;0.233177985624513;0.0276934327001115;0.000232670765719012;6.67538218020217e-05;7.98621908652810e-05];
B_llh = B(1:57);

%K-nearest neighbor regression
knn = fitcknn(X,y,'NumNeighbors',1,'Standardize',1);
y_pred_knn = predict(knn, X);
Accuracy_knn = mean(double(y_pred_knn == y)) * 100;

% Plot Estimated coefficients for regression method on a spam data
factor = 1:57 ;
factor_t = transpose(factor);
figure()
plot(factor_t, B_ols, 'ro', factor_t, B_ridge, 'go' , factor_t, B_lasso_mean_T, 'yo', factor_t, B_pcr,'bo', factor_t, B_pls, 'mo',factor_t, B_llh,'ko' )
hold on
plot(B_ols,'r','LineWidth',0.7)
plot(B_ridge, 'g','LineWidth', 0.7)
plot(B_lasso_mean_T, 'y','LineWidth', 0.7)
plot(B_pcr, 'b','LineWidth', 0.7)
plot(B_pls, 'm','LineWidth', 0.7)
plot(B_llh, 'k','LineWidth', 0.7)
xlabel('Factor')
ylabel('Regression Coefficients')
title('Estimated coefficients for regression method on a spam data')
legend("OLS","Ridge","Lasso","PCR", "PLS","Loglikelihood");  



