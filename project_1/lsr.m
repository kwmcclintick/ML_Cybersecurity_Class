% AUTHOR: KWM
% Comment: LSR works is innapropriate for this problem for two reasons:
% 1) its predictions are continuous while the datas labels are discrete
% 2) localization problems are typically not linearly related
% As a result, I would have used a non-linear softmax regression solution,
% but MATLAB has good decision tree support so I went with that instead

week1 = load('week1.mat'); week1 = week1.week1;
week2 = load('week2.mat'); week2 = week2.week2;
week3 = load('week3.mat'); week3 = week3.week3;
week4 = load('week4.mat'); week4 = week4.week4;

X_tr = [week1; week2; week3; week4];

y = X_tr(:,5);
X_tr = X_tr(:,1:4);

% w = inv(X_tr'*X_tr)*X_tr'*y;
Mdl = fitctree(X_tr,y);

week5 = load('week5.mat'); week5 = week5.week5;

y_te = week5(:,5);
X_te = week5(:,1:4);

% y_hat = X_te*w;
y_hat = predict(Mdl,X_te);
fMSE = 1 / (2 * length(y_te)) * sum((y_hat - y_te).^2)
hold on;
plot(y_hat,'+')
plot(y_te,'^')
legend('predictions','truth')
