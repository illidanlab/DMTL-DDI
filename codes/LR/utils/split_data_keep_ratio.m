function [trainX,trainY,validX,validY, testX,testY,indices] = split_data_keep_ratio(data,label,ratio, seed)
% THIS script split the data into training testing and validation set.
% Don't not normalize the data, KEEP the positve/negative number of
% samples same in the training and testing
% INPUT: data: cells K x 1; label: cells K x 1; ratio = [0.5,0.3]
% percentage of training data and validation data. the rest of it is the
% percentage of testing data
% OUTPUT: struct: trainX, trainY; validX, validY; testX, testY

[~, K] = size(data); % number of tasks.

 

trainX = cell(1,K);
trainY =  cell(1,K);

validX =  cell(1,K);
validY =  cell(1,K);

testX  =  cell(1,K);
testY  =  cell(1,K);

indices = cell(1,K);

% trainallX = cell(1,K);
% trainallY = cell(1,K);
for i = 1:K
   
    X = data{i};
    Y = label{i};
    
    % find positive samples
    pos_inds = find(Y==1);  
    neg_inds = find(Y==-1);
    
    [pos_train_inds, pos_valid_inds, pos_test_inds] = split_indices(pos_inds, ratio, i);
    [neg_train_inds, neg_valid_inds, neg_test_inds] = split_indices(neg_inds, ratio, i);
       
    train_index = [pos_train_inds; neg_train_inds];
    test_index = [pos_test_inds; neg_test_inds];
    valid_index = [pos_valid_inds; neg_valid_inds];
    
    rng(seed)
    train_index = train_index(randperm(length(train_index)));
    rng(1 + seed)
    test_index = test_index(randperm(length(test_index)));
    rng(2 + seed)
    valid_index = valid_index(randperm(length(valid_index)));
    
    train_x = X(train_index,:);
    train_y = Y(train_index,:);
     
    [train_X,train_Y,train_index_sampled] = negative_sampling_byindices(train_x, train_y, train_index);
    
    trainX{i} = train_X;
    trainY{i} = train_Y;
    
    validX{i} = X(valid_index,:);
    validY{i} = Y(valid_index,:);
    
    testX{i}  = X(test_index,:);
    testY{i}  = Y(test_index,:);
   
    indices{i} = struct('train_index',train_index_sampled ,'test_index',test_index, 'valid_index',valid_index);
end
end