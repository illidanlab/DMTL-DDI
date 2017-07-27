function [AUC, accuracy,f_measure, p1] = evaluate_acc(X, y, w, c)

x = X{1};
y = y{1};

aa = (x*w + c);

p1 = 1./(1 + exp(-1*aa)); % the probability that classify the instance to 1.


y_pred = p1;

y_pred(y_pred>=0.5) = 1;

y_pred(y_pred<0.5) = -1;

f_measure = Evaluate_f1(y, y_pred);


err = y_pred - y;

num_correct = sum(err(:)==0);

accuracy = num_correct/length(y);

if (length(unique(y)) == 2)
    [X_axis,Y_axis,T,AUC] = perfcurve(y,p1,1);
else
    AUC = -1;
end


end