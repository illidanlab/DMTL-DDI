function [trainX,trainY,validX,validY, testX,testY] = split_data_zscoreX(data,label,ratio)
% THIS script split the data into training testing and validation set. ONLY
% normalize X for binary input
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

% trainallX = cell(1,K);
% trainallY = cell(1,K);
for i = 1:K
   
    X = data{i};
    Y = label{i};
    
    % normalize X 
    X = zscore(X);
 
    
    [num, feature] = size(X);
    rng(i)
    index = randperm(num);
    
    train_num = round(num*ratio(1));
    valid_num = round(num*ratio(2));
    
    train_index = index(1:train_num);
    valid_index = index(train_num+1:train_num+valid_num);
    test_index = index(train_num+valid_num+1:end);
    
%     trainall_index = index(1:train_num+valid_num);

%     trainallX{i} = X(trainall_index,:);
%     Yi           = Y(trainall_index,:);
%     [Yi_zscored,Yimean,Yistdev] = zscore(Yi); % normalize label in training set
    
%     trainallY{i} = Yi_zscored;
    
    train_x = X(train_index,:);
    train_y = Y(train_index,:);
    
    [train_X,train_Y] = negative_sampling(train_x,train_y);
    
    trainX{i} = train_X;
    trainY{i} = train_Y;
    
    validX{i} = X(valid_index,:);
    validY{i} = Y(valid_index,:);
    
    testX{i}  = X(test_index,:);
    testY{i}  = Y(test_index,:);
   
    
end
end