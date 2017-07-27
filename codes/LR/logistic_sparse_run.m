timeflag = strrep(datestr(datetime),' ','');
disp(timeflag);
timeflag = strrep(timeflag,':','-');
timeflag = timeflag(1:end-3);

%%% This scripts run logistic regression for multiple envents ids
%%% The splitting is keep pos/neg ratio in dataset the same.

%% Load data
addpath(genpath('./'))
data_dir = '../../../../AllData/Fei-DrugData/ddi_prep/';
filename = strcat('eventall_featlabel_clean_sym.mat');

data_dict = load(strcat(data_dir, filename));

features = {data_dict.features};
labels = {data_dict.labels};
ratio = [0.8, 0.1];
seed = 1024; % fold 2: Mar 

[nn, K] = size(labels{1});

results = cell(K,1);
configure_file;


tunedParas = lr_sparse.tunedParas;

% parallel computing parameters  %%%%%%%%%%%%%%%%%
lenPara = 1;
paraNames = fieldnames(tunedParas);
num_paras = length(paraNames);
len_paras = zeros(num_paras,1);
dividend  = ones(num_paras,1);   

para_i = cell(num_paras,1);
for i = 1:num_paras
    para_i{i} = getfield(tunedParas, paraNames{i});
    lenPara = lenPara*length(para_i{i});
    len_paras(i) = length(para_i{i});  % lenght of each parameter array
    
    if i ~= num_paras
        for jj = i+1:num_paras
            dividend(i) = dividend(i) * length(getfield(tunedParas, paraNames{jj}));
        end
    else
        dividend(i) = len_paras(i);
    end
end


%%% zscore the feature set at the begining:
mkdir(strcat(data_dir,'matlab_results/', timeflag, '/'));


% taskids = [446,354,805,1247]; % 
parfor ii = 1:K

% task_id = taskids(ii); % for sepcific tasks
task_id = ii; % for all tasks


label = {labels{1}(:,task_id)};


% negative sampling and spliting  %%% NOTICE indices only for one task
[train_x,train_y,valid_x,valid_y, test_x,test_y, indices] = split_data_keep_ratio(features,label,ratio,seed);

num_pos_sample = length(find(label{1}==1));
num_pos_sample_train = length(find(train_y{1}==1));
num_pos_sample_valid = length(find(valid_y{1}==1));
num_pos_sample_test = length(find(test_y{1}==1));
% fprintf('start training')


Ws = cell(lenPara,1);
Cs = cell(lenPara,1);
funcVals = cell(lenPara,1);
accuracys = zeros(lenPara,1);
aucs = zeros(lenPara, 1);
f_measures = zeros(lenPara, 1);

for i = 1:lenPara
        paras_indexs = zeros(num_paras,1); % the real index in each array
        for j = 1:num_paras
            if j == num_paras
                paras_indexs(j) = mod(i,len_paras(j));

            else
                temp = ceil(i/dividend(j));
                paras_indexs(j) = mod(temp,len_paras(j));
            end 
            if paras_indexs(j)== 0
                    paras_indexs(j) = len_paras(j);
            end
        end


        parameters = struct();
        for jj = 1:num_paras
            para_array = para_i{jj}; % current parameters arrays.
            parameters.(paraNames{jj}) = para_array(paras_indexs(jj)) ;
        end

        opts = [];
        opts.maxIter = 1000;
        opts.tFlag = 1; % maxIterations
        opts.tol = 1e-6;
        rho1 = parameters.(paraNames{1});
        opts.rho_L2 = parameters.(paraNames{2}); 
        
        
        [W, C, funcVal] = Logistic_Lasso(train_x, train_y, rho1, opts);  
%         W = rand(1232,1);
%         C = rand(1,1);
%         funcVal = 1;
%         disp(sprintf('rho1 %.4f, rho2 %.4f', rho1, opts.rho_L2)); checked

        
        [AUC, accuracy, f_measure, ~] = evaluate_acc(valid_x, valid_y, W, C); % if AUC = -1 there is only one class in testing.
        Ws{i} = W;
        Cs{i} = C;
        funcVals{i} = funcVal;
        f_measures(i,1) =  f_measure;
        aucs(i,1) = AUC;
        accuracys(i,1) = accuracy;
end

[f1, index_f1] = max(f_measures);

[auc, index_auc] = max(aucs);
funcval = funcVals{index_auc};

Wauc = Ws{index_auc};
Cauc = Cs{index_auc};
Wf1 = Ws{index_f1};
Cf1 = Cs{index_f1};

[auc_test, ~, ~, ~] = evaluate_acc(test_x, test_y, Wauc, Cauc);
[auc_train, ~, ~, ~] = evaluate_acc(train_x, train_y, Wauc, Cauc);
[auc_valid, ~, ~, ~] = evaluate_acc(valid_x, valid_y, Wauc, Cauc);

[~, ~, f1_test, ~] = evaluate_acc(test_x, test_y, Wf1, Cf1);
[~, ~, f1_train, ~] = evaluate_acc(train_x, train_y, Wf1, Cf1);
[~, ~, f1_valid, ~] = evaluate_acc(valid_x, valid_y, Wf1, Cf1);

results{ii} = {accuracys, aucs,Ws,Cs, index_f1, index_auc, ...
                f1_test, f1_train, f1_valid,...
                auc_test,auc_train,auc_valid,...
               num_pos_sample_train, num_pos_sample_test,num_pos_sample_valid,num_pos_sample,indices{1}, task_id};

           
% print results
disp(sprintf(strcat('The ', int2str(task_id),...
         'th tasks: f1 {test: %.4f, train: %.4f, valid: %.4f},',...
         'Auc {test: %.4f, train: %.4f, valid: %.4f}',... 
         'index: %d,index2: %d, pos train: %d,pos test: %d,pos valid: %d, pos all: %d'),...
         f1_test, f1_train, f1_valid,...
         auc_test, auc_train, auc_valid,...
         index_f1,index_auc,num_pos_sample_train, num_pos_sample_test,num_pos_sample_valid,num_pos_sample));

fig = plot(funcval);
saveas(fig,strcat(data_dir,'matlab_results/', timeflag,sprintf('/task%d_objvalue_converge.png',task_id)));



fileID = fopen(strcat(data_dir,'matlab_results/',timeflag, '/temp_all/', strrep(filename,'.mat',''), timeflag, 'task', int2str(task_id), '.txt'),'w');
fprintf(fileID,...
        sprintf(strcat('The ', int2str(task_id),'th tasks: f1 {test: %.4f, train: %.4f, valid: %.4f},',...
                       'Auc {test: %.4f, train: %.4f, valid: %.4f}',... 
                       'index: %d,index2: %d, pos train: %d,pos test: %d,pos valid: %d, pos all: %d'),...
                       f1_test, f1_train, f1_valid,...
                       auc_test,auc_train,auc_valid,...
                       index_f1,index_auc,num_pos_sample_train, num_pos_sample_test,num_pos_sample_valid,num_pos_sample));
fclose(fileID);
end
results_description = strcat('accuracys, aucs,Ws,Cs, index_f1, index2',...
                             'f1_test, f1_train, f1_valid,',...
                             'auc_test,auc_train,auc_valid,',...
                             'num_pos_sample_train, num_pos_sample_test,num_pos_sample_valid,num_pos_samplem, spliting_indices, task_id');
save(strcat(data_dir, 'matlab_results/',timeflag, '/', strrep(filename,'.mat',''), timeflag,'.mat'),'results','results_description', 'ratio','lr_sparse');
