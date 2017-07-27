function [trainX,trainY, train_indices] = negative_sampling_byindices(trainX,trainY, train_index)
% Sampling the negative/pos samples to make sure the negative sample and
% positive sample are 1:1, by return the indices. 
% postive label and negative label denotes as -1, 1
% input: nxd matrix trainX, nx1 matrix trainY


neg_indices = find(trainY==-1);
pos_indices = find(trainY==1);

len_neg = length(neg_indices);
len_pos = length(pos_indices);


selected_inds = [];
if (len_neg > len_pos)
    rng(1024)
    index = randperm(len_neg);
    neg_indices_sampled = neg_indices(index(1:len_pos));
    
    selected_inds = [neg_indices_sampled; pos_indices];
    trainY = trainY(selected_inds);
    trainX = trainX(selected_inds,:);
elseif (len_pos > len_neg)
     rng(1024)
     index = randperm(len_pos);
     pos_indices_sampled = pos_indices(index(1:len_neg));
     selected_inds = [pos_indices_sampled, neg_indices];
     trainY = trainY(selected_inds);
     trainX = trainX(selected_inds,:);
end
[n_samp, ~] = size(trainX);
rng(1024)
indices = randperm(n_samp);

trainX = trainX(indices,:);
trainY = trainY(indices,:); 
train_indices = train_index(selected_inds,:); 
train_indices = train_indices(indices, :); % rand permutation

end