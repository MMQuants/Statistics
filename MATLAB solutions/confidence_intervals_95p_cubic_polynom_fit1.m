% Raw data
years = [1959 , 1960 , 1961 , 1962 , 1963 , 1964 , 1965 , 1966 , 1967 , 1968 , 1969];
x = transpose(years);
population = [4835 ,4970 , 5085 , 5160 , 5310 , 5260 , 5235 , 5255 , 5235 , 5210 , 5175];
y = transpose(population);
% Rescale years
M = mean(x);
xnorm = x - M;
% Generate regression , construct confidence intervals
p_fit = polyfit(xnorm,y , 3);
mdl = fitlm(xnorm, y, 3);
y_pred = predict(mdl,xnorm);

beta0 = p_fit;
modelfun = @(b,xnorm)(b(1)+b(2)*xnorm + b(3)*xnorm.^2+ b(4)*xnorm.^3);
[beta,R,J,CovB,MSE] = nlinfit(xnorm,y,modelfun,beta0);
[Ypred,delta] = nlpredci(modelfun,xnorm,beta,R,'Covar',CovB);
lower = Ypred - delta;
upper = Ypred + delta;
%xrange = min(xnorm):.01:max(xnorm);
figure()
plot(xnorm,y,'ko') % observed data
hold on
plot(xnorm,Ypred,'k','LineWidth',2)
plot(xnorm,[lower;upper],'r--','LineWidth',1.5)
