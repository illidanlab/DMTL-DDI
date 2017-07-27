function [train_inds, valid_inds, test_inds] = split_indices(total_inds, ratio, seed)
    

    num = length(total_inds);
    
    rng(seed)
    index = randperm(num);
    
    train_num = round(num*ratio(1));
    valid_num = round(num*ratio(2));
    
    train_inds = total_inds(index(1:train_num));
    valid_inds = total_inds(index(train_num+1:train_num+valid_num));
    test_inds = total_inds(index(train_num+valid_num+1:end));

end