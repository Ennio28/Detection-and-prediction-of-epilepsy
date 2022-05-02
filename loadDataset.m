function [P_train,T_train, P_test, T_test] = loadDataset(dataset_idx, balance)

    trainRatio = 0.75;
    
    % Load Patient dataset
    if dataset_idx == 1
        load('54802.mat');
    elseif dataset_idx == 2
        load('112502.mat');
    end
    
    %Indexing the target.
    % 0 (non-ictal) and 1 (ictal) dataset to 4 classes (interictal, preictal 
    % ictal and posictal)
    T = indexing_target(Trg);
       
    % Assign features to P
    P = FeatVectSel;
    
    % Last index of the last posictal instance
    last_ictal = find(T(:,3) == 1);
    last_index = last_ictal(end) + 1;
        
    % Keep only data before that instance
    P = P(1:last_index,:);
    Q = length(P);
    testRatio =  1 - trainRatio;
    [trainInd,testInd] = divideblock(Q, trainRatio, testRatio);
        
    % Define the test set
    P_test = P(testInd,:)';
    T_test = T(testInd,:)';
        
    if balance == 0    
        % Define the training set
        P_train = P(trainInd,:)';
        T_train = T(trainInd,:)';
        return;
    end
    
    % Vector that indicated the begining and end of each seizure (1 and -1)
    % by checking the diffence between index i+1 and i
    seizures = diff(Trg);
    % Indexes with the first pre-ictal (2) of each seizure
    start_preictal = find(seizures == 1) - 900;
    % Indexes with the last post-ictal (4) of each seizure
    end_posictal = find(seizures == -1) + 301;
    
    total_seizures = length(start_preictal);
    
    % Number of seizures present in the training data
    num_seizures_training_data = round(total_seizures * trainRatio);
    
    % Vector with preictal (2), ictal (3) and postictal (4) for the
    % training set
    seizures_idxs = [];
    for i = 1 : num_seizures_training_data
        seizures_idxs = [seizures_idxs start_preictal(i):1:end_posictal(i)];
    end
    
    % Balance dataset : The number of inter_ictal (1) class instances is equal
    % to the sum of all other classes;;
    total_interictal = length(seizures_idxs);
    
    % Choose inter_ictal indexes in a equal number to the total ones
    interictal_idxs = find(T == 1)';
    interictal_idxs = interictal_idxs(1:total_interictal);
    
    % Join interinctal (1) class and the remaining classes (2,3,4 -
    % seizure)
    training_idxs = horzcat(interictal_idxs,seizures_idxs);
    training_idxs = sort(training_idxs);
    
    % Define the training set
    P_train = P(training_idxs,:)';
    T_train = T(training_idxs,:)';   
    
%end

