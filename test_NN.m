function [inter_ictal, pre_ictal, ictal] = test_NN(nn_type, net, p_matrix, t_matrix, group)
    %Prepare structures to receive the false and true negatives / positives
    %Structure for class 1 (inter ictal)
    inter_ictal.true_positive = 0;
    inter_ictal.true_negative = 0;
    inter_ictal.false_positive = 0;
    inter_ictal.false_negative = 0;
    
    %Structure for class 2 (pre ictal)
    pre_ictal.true_positive = 0;
    pre_ictal.true_negative = 0;
    pre_ictal.false_positive = 0;
    pre_ictal.false_negative = 0;
    
    %Structure for class 3 (ictal)
    ictal.true_positive = 0;
    ictal.true_negative = 0;
    ictal.false_positive = 0;
    ictal.false_negative = 0;
    
    if strcmp(nn_type, 'autoencoder')
        result = net(p_matrix);
        plotconfusion(t_matrix, result);
    end
    
    if (strcmp(nn_type, 'CNN')) || (strcmp(nn_type, 'LSTM'))
        if strcmp(nn_type, 'LSTM')
            p_matrix = num2cell(p_matrix, 1);
            result = classify(net, p_matrix, 'MiniBatchSize', 29);
            
        elseif strcmp(nn_type, 'CNN')
            T_test= zeros(1, length(t_matrix)); 
            T_test(find(t_matrix(1,:) == 1)) = 1;
            T_test(find(t_matrix(2,:) == 1)) = 2;
            T_test(find(t_matrix(3,:) == 1)) = 3;
            T_test = categorical(T_test)';
        
            CNN_PTrain = p_matrix(1:29,1:29,1);
            CNN_TTrain = ones(1,length(T_test));
            i = 30;
            img_number = 2;
            while i+28 <= length(p_matrix)
                count = 0;
                for j = i:i+28
                    if T_test(j) == T_test(i)
                        count = count+1;
                    end
                end   
    
                if (count == 29) && (i+28 <= length(p_matrix))
                    part = p_matrix(1:29,i:i+28,1);
                    CNN_PTrain (:,:,:,img_number) = part;
                    CNN_TTrain(img_number) = T_test(i);
                    img_number = img_number+1;
                end   
                i = i+count;
            end
            result = classify(net, CNN_PTrain, 'MiniBatchSize', 29);
            
            if size(CNN_TTrain) ~= img_number
                CNN_TTrain = CNN_TTrain(1:img_number-1);
            end
            
            t_matrix = CNN_TTrain;
            
            result_t_matrix = zeros(length(t_matrix), 3);
            for i = 1 : length(t_matrix)
                result_t_matrix(i, t_matrix(i)) = 1;
            end
        
            t_matrix = result_t_matrix';
        
        end
        
        result_matrix = zeros(length(result), 3);
        for i = 1 : length(result)
            result_matrix(i, result(i)) = 1;
        end
        
        result = result_matrix';
    else
        %Run the input matrix on the trained NN
        result = sim(net, p_matrix);
    end

    %Prepare a matrix to receive the 'ones' for each class (as the outcome
    %of the NN training)
    [~, index] = max(result);
    result = zeros(3, length(result));
    
    for i=1:length(result)
        result(index(i), i) = 1;
    end
    
    if group == true
        for i=1:length(result)-10
            count = 1;
            for j=1:9
                if result(:,i) == result(:,i+j)
                    count = count + 1;
                end               
            end
            if count < 9
                result(:,i) = [1;0;0];
            end
        end
        group_ictal = diff(result(3,:));
        group_pre_ictal = diff(result(2,:));
        
        ictal_count = 0;
        pre_ictal_count = 0;
        for i=1:length(result)-1
            if group_ictal(i) == 1
                ictal_count = ictal_count + 1;
            elseif group_pre_ictal(i) == 1
                pre_ictal_count = pre_ictal_count + 1;
            end
        end
        disp(ictal_count)
        disp(pre_ictal_count)
    end
    
    for i=1:3       
        if i==1
            for j=1:length(result)
                if(result(i,j) == t_matrix(i,j))
                    if(result(i,j) == 1)
                        inter_ictal.true_positive = inter_ictal.true_positive + 1;
                    else
                        inter_ictal.true_negative = inter_ictal.true_negative + 1; 
                    end
                else
                    if(result(i,j) == 1)
                        inter_ictal.false_positive = inter_ictal.false_positive + 1;
                    else
                        inter_ictal.false_negative = inter_ictal.false_negative + 1; 
                    end
                end
            end
            
        elseif i==2
            for j=1:length(result)
                if(result(i,j) == t_matrix(i,j))
                    if(result(i,j) == 1)
                        pre_ictal.true_positive = pre_ictal.true_positive + 1;
                    else
                        pre_ictal.true_negative = pre_ictal.true_negative + 1; 
                    end
                else
                    if(result(i,j) == 1)
                        pre_ictal.false_positive = pre_ictal.false_positive + 1;
                    else
                        pre_ictal.false_negative = pre_ictal.false_negative + 1; 
                    end
                end
            end
            
        elseif i==3
            for j=1:length(result)
                if(result(i,j) == t_matrix(i,j))
                    if(result(i,j) == 1)
                        ictal.true_positive = ictal.true_positive + 1;
                    else
                        ictal.true_negative = ictal.true_negative + 1; 
                    end
                else
                    if(result(i,j) == 1)
                        ictal.false_positive = ictal.false_positive + 1;
                    else
                        ictal.false_negative = ictal.false_negative + 1; 
                    end
                end
            end
        end
    end