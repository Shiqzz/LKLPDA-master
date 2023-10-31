clear all
addpath('Datasets');
load piRNA_Disease.mat
load reshape_data.mat

Wpp2 = PiRNA_seq;
Wdd2 = Disease_mesh;
Wdd3 = Disease_gene;

pidi=A;
Wpd=pidi;
Wdp = Wpd';
lamda1=2000;
lamda2=2000;



%Storage piRNA fusion similarity matrix in five-fold cross validation
piRNA_allsi = {};
disease_allsi = {};
%Storage piRNA and disease similarity matrix splicing
feature = {};
% Store each training set and test set,
train_data ={};
test_data ={};

%%%%%%%%%%%%%% set the CV parameters %%%%%%%%%%   
nCV = 5;        

PoPMat = find(Wpd==1);
NumAs = length(PoPMat);
[dd,dp] = size(Wpd);


% % Randomly generate negative samples that are equal to the positive samples
% random_Negative_sample = [];
% while numel(random_Negative_sample) < length(PoPMat)
%     random_Negative= randi([1, dd*dp]);
%     if ~ismember(random_Negative, PoPMat) && ~ismember(random_Negative, random_Negative_sample)
%         random_Negative_sample = [random_Negative_sample, random_Negative];
%     end
% end
% 
% %Splice the positive samples and negative samples together and calculate the overall number
% all_poMat = [PoPMat;random_Negative_sample'];
% all_NumAs = length(all_poMat);
% T_NumAs = ceil(all_NumAs/nCV)*nCV;
% 
% tic
% random_indices = randperm(all_NumAs);
% random_indices(all_NumAs+1:T_NumAs) = 0;
% 
% % Randomly arrange the positive and negative samples into five copies
% Indices_groups = reshape(random_indices(1:floor(length(random_indices)/nCV)*nCV), nCV, floor(length(random_indices)/nCV));
% 
% ass_num = 1;
% tic


for num = 1:nCV
    num;
%   rand('state', num); %#ok<RAND>
    rng(num);
  
    G_TestIds = Indices_groups(num,:);
    G_TestIds(G_TestIds==0) = [];
%%%%%%%%%% Tfnum indicates the number of elements in each group %%%%%%%%%%
        Tfnum = length(G_TestIds);
        TestIds = all_poMat(G_TestIds);
        P_TMat = Wpd;
        P_TEst = Wpd;      
        P_TMat(TestIds) = 0;
  
        %Use the training data set to generate piRNA and disease Gaussian kernel function
       [Wpp1,Wdd1] = Rbf_kernel(P_TMat);
        % Form a three-dimensional similarity nuclear matrix of piRNA
         K1(:,:,1) = Wpp1;
         K1(:,:,2) = Wpp2;
        % Form a three-dimensional similarity kernel matrix of the disease
        K2(:,:,1) = Wdd1;  
        K2(:,:,2) = Wdd2;   
        K2(:,:,3) = Wdd3;  
        Wpp = K1;
        Wdd = K2;
        Wpp(Wpp==0) =1e-12; 
        Wdd(Wdd==0) =1e-12; 
   
        %LRFKL
        [weight_up] = LR_FKL(Wpp,P_TMat,2,lamda1,lamda2);
        PS = combine_kernels(weight_up, Wpp);
        [weight_ud] = LR_FKL(Wdd,P_TMat,1,lamda1,lamda2);
        DS = combine_kernels(weight_ud, Wdd);


        
        piRNA_allsi{num}=PS;
        disease_allsi{num}=DS;       
        % 创建逻辑索引，将不想要的数值设为 true
        excludeMask = ismember(all_poMat, TestIds);
        train_data{num}= all_poMat(~excludeMask);
        test_data{num} = TestIds;
        all_data{1}=all_poMat;
 end 
    


    train_label = label_treatment(train_data,Wpd,PoPMat);
    test_label = label_treatment(test_data,Wpd,PoPMat);
    all_label = label_treatment(all_data,Wpd,PoPMat);

    test_feature = feature_label(test_label,piRNA_allsi,disease_allsi);
    train_feature = feature_label(train_label,piRNA_allsi,disease_allsi);
    

%     folderPath = '../Datasets/five-fold';  
    cell_csv(train_label,'train_label',folderPath);
    cell_csv(test_label,'test_label',folderPath);
    cell_csv(all_label,'all_label',folderPath);
    cell_csv(test_feature,'test_feature',folderPath);
    cell_csv(train_feature,'train_feature',folderPath);



    function [csv_Object]=cell_csv(cell_Object,csv_name,folderPath)

        for i = 1:numel(cell_Object)
       
            if contains(csv_name, 'label')&& ~contains(csv_name, 'all')
                 subVariable = cell_Object{i}(:,3);
            else
                 subVariable = cell_Object{i};
            end
            tableName = [csv_name, num2str(i), '.csv'];
            filePath=fullfile(folderPath, tableName);  % 构建完整的文件路径
            writematrix(subVariable, filePath, 'Delimiter', ',');
        end
    end
    
    
    function [feature_Object]=feature_label(label,piRNA_allsi,disease_allsi)
           
                feature_Object = cell(size(label));
                for i = 1:numel(label)
                   
                    new_feature_data = zeros(size(label{i},1),size(piRNA_allsi{i},2)+size(disease_allsi{i},2));
                        for j = 1:size(label{i},1)

                           disease_Location = label{i}(j,1);                  
                           piRNA_loction = label{i}(j,2);
                           new_feature_data(j,1:523) = [piRNA_allsi{i}(piRNA_loction,1:size(piRNA_allsi{1})),disease_allsi{i}(disease_Location,1:size(disease_allsi{1}))];
                        end
                    feature_Object{i}=new_feature_data;
                end
    
    
    end

    function [newCellObject]=label_treatment(CellObject,Wpd,PoPMat)
        newCellObject = cell(size(CellObject));

        for i = 1:numel(CellObject)

                dataSize = size(CellObject{i});
                newSubVariable = zeros(dataSize(1),dataSize(2)+2);

                for j = 1:numel(CellObject{i})
                    [rows, cols] = ind2sub(size(Wpd), CellObject{i}(j));
                    if ismember(CellObject{i}(j), PoPMat)
                       dataWithIndices =[rows, cols, 1];
                    else 
                       dataWithIndices =[rows, cols, 0];
                    end
                    newSubVariable(j,1:dataSize(2)+2)=dataWithIndices;
                end

                newCellObject{i}=newSubVariable;
        end
    end
    function [w] = LR_FKL(Kernels_list,adjmat,dim,lamda1,lamda2)
        % adjmat : binary adjacency matrix
        % dim    : dimension (1 - rows, 2 - cols)
        % lamda: Regularized item (2000)
        num_kernels = size(Kernels_list,3);
        %r_lamda = 2000;
        weight_v = zeros(num_kernels,1);
        
        y = adjmat;
        if dim == 1
                 ga = LRR(y',lamda1);
               
        else
                 ga = LRR(y,lamda1);
        end
        
        
        N_U = size(y,dim);
        l=ones(N_U,1);
        U = eye(N_U) - (l*l')/N_U;
        
        M = zeros(num_kernels,num_kernels);
        
        for i=1:num_kernels
	        for j=1:num_kernels
		        kk1 = Kernels_list(:,:,i);
		        kk2 = Kernels_list(:,:,j);
		        mm = trace(kk1'*kk2);
		        M(i,j) = mm;
	        end
        end
        
        a = zeros(num_kernels,1);
        
        for i=1:num_kernels
        
	        kk = Kernels_list(:,:,i);
	        aa = trace(ga'*kk);
	        a(i) = aa;
        end
        
        v = randn(num_kernels,1);
        M_A = (M + lamda2*eye(num_kernels));
        cvx_begin
            variable v(num_kernels,1);
            minimize( v'*M_A*v - 2*a'*v );
	        v >= 0;
	        sum(v)==1;
        cvx_end
        
        %w = v /sum(v);
        %w = v /norm(v,2);
        w = v;
    end
    function result = combine_kernels(weights, kernels)
        % length of weights should be equal to length of matrices
        n = length(weights);
        result = zeros(size(kernels(:,:,1)));    
        
        for i=1:n
            result = result + weights(i) * kernels(:,:,i);
        end
    end
    function [Z,E] = LRR(X,lambda)
        Q = orth(X');
        A = X*Q;
        [Z,E] = lrra(X,A,lambda);
        Z = Q*Z;
    end

