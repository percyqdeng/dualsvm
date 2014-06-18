load('../../dataset/usps/usps_all.mat');
pos_ind = 4;

x = [];
y = [];
for i = 1:10
    x = [x;data(:,:,i)'];
    n = size(data(:,:,i)',1);
    if i-1 == pos_ind
        y = [y; ones(n,1)];
    else
        y = [y; -1 * ones(n,1)];
    end
    
end
x = double(x);
kfold = 5
indices = crossvalind('Kfold', length(y), kfold);
n_neighbors = [1,2,4,5,8,10,15,20,27,36,42,55,60];
test_err = zeros(length(n_neighbors), kfold);
train_err = zeros(length(n_neighbors), kfold);
for i = 1:kfold
   x_train = x(indices~=i);
   y_train = y(indices~=i);
   x_test = x(indices==i);
   y_test = y(indices==i);
   disp(['kfold: ',num2str(i)]);
   for k = 1:length(n_neighbors)
       pred = knnclassify(x_test, x_train, y_train, n_neighbors(k));
       test_err(k,i) = mean(y_test ~= pred);
       pred = knnclassify(x_train, x_train, y_train, n_neighbors(k));
       train_err(k,i) = mean(y_train ~= pred);
   end
    
end

avg_train_err = mean(train_err,2);
avg_test_err = mean(test_err, 2);

plot(n_neighbors, avg_train_err, 'rx-');
hold on;
plot(n_neighbors, avg_test_err, 'bo-');
legend('train err', 'test err');