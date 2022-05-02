function net = train_NN(nn_type, p_matrix, t_matrix, type, hiddensize1, hiddensize2)
    if (strcmp(nn_type, 'multilayer')) || (strcmp(nn_type, 'Multilayer with Delays'))
        weight_penalty = ones(1, length(t_matrix));
    
        interictal_idxs = find(t_matrix(1,:) == 1);
        interictal_total = length(interictal_idxs);
        preictal_idxs = find(t_matrix(2,:) == 1);
        preictal_total = length(preictal_idxs);
        ictal_idxs = find(t_matrix(3,:) == 1);
        ictal_total = length(ictal_idxs);
    
        if strcmp(type, 'prediction')
            error = (interictal_total / preictal_total) * 2;
            weight_penalty(preictal_idxs) = error;
        elseif strcmp (type, 'detection')
            error = (interictal_total / ictal_total) * 18;
            weight_penalty(ictal_idxs) = error;
        end
    end
    
    if strcmp(nn_type, 'multilayer')
        net = feedforwardnet(25);
        
        net.divideFcn = 'divideblock';
        net.trainFcn = 'trainlm';
        net.trainParam.epochs = 1000;      
        net.trainParam.max_fail = 5000;
    
        % Train division
        net.divideParam.trainRatio = 0.80;
        net.divideParam.valRatio = 0.20;
        net.divideParam.testRatio = 0.0;
        
        init(net);
    
        net = train(net, p_matrix, t_matrix, [], [], weight_penalty);
        
        save('multilayer.mat', 'net');
        
    elseif strcmp(nn_type, 'Multilayer with Delays')
        net = layrecnet(1:2,10);
        net.divideFcn = 'divideind';
        net.trainFcn = 'trainlm';
        [net.divideParam.trainInd,~,~] = divideind(length(p_matrix),1:length(p_matrix),[],[]);
    
        net.trainParam.epochs = 1000;
        net = train(net, p_matrix, t_matrix, [],[], weight_penalty);
        
        save('multilayerdelay.mat', 'net');
        
    elseif strcmp(nn_type, 'CNN')
        layers = [
            imageInputLayer([29 29 1])
            convolution2dLayer(5,20)
            reluLayer()
            maxPooling2dLayer(2,'Stride',2)
            fullyConnectedLayer(3)
            softmaxLayer
            classificationLayer
        ];
        
        options = trainingOptions('sgdm', ...
            'MaxEpochs',1000, ...
            'InitialLearnRate',1e-4, ...
            'Shuffle','every-epoch', ...
            'Plots','training-progress', ...
            'Verbose',false);
    
        T_train= zeros(1, length(t_matrix)); 
        T_train(find(t_matrix(1,:) == 1)) = 1;
        T_train(find(t_matrix(2,:) == 1)) = 2;
        T_train(find(t_matrix(3,:) == 1)) = 3;
        T_train = categorical(T_train)';
        
        CNN_PTrain = p_matrix(1:29,1:29,1);
        CNN_TTrain = ones(1,length(T_train));
        i = 30;
        image_nummber = 2;
        while i+28 <= size(p_matrix,2)
            count = 0;
            for j = i:i+28
                if T_train(j) == T_train(i)
                    count = count+1;
                end
            end   
    
            if (count == 29) && (i+28 <= size(p_matrix,2))
                part = p_matrix(1:29,i:i+28,1);
                CNN_PTrain (:,:,:,image_nummber) = part;
                CNN_TTrain(image_nummber) = T_train(i);
                image_nummber = image_nummber+1;
            end
    
            i = i+count;
        end    

        if size(CNN_TTrain) ~= image_nummber
            CNN_TTrain=CNN_TTrain(1:image_nummber-1);
        end
        
        CNN_TTrain = categorical(CNN_TTrain)';
        
        net = trainNetwork(CNN_PTrain, CNN_TTrain, layers, options);
        
        save('cnn.mat', 'net');
        
    elseif strcmp(nn_type, 'LSTM')
        numFeatures = 29;
        numHiddenUnits = 75;
        numClasses = 3;
        layers = [
            sequenceInputLayer(numFeatures)
            lstmLayer(numHiddenUnits,'OutputMode','last')
            fullyConnectedLayer(numClasses)
            softmaxLayer
            classificationLayer
        ];

        options = trainingOptions('adam', ...%sgdm
            'ExecutionEnvironment','auto', ...
            'GradientThreshold',1, ...
            'MaxEpochs',1000, ...
            'MiniBatchSize',29, ...
            'SequenceLength','longest', ...
            'Shuffle','never', ...
            'Verbose',0, ...
            'Plots','training-progress');

        p_matrix = num2cell(p_matrix,1);
    
        T_train= zeros(1, length(t_matrix)); 
        T_train(find(t_matrix(1,:) == 1)) = 1;
        T_train(find(t_matrix(2,:) == 1)) = 2;
        T_train(find(t_matrix(3,:) == 1)) = 3;
        T_train = categorical(T_train)';
 
        net = trainNetwork(p_matrix, T_train, layers, options);
        
        save('lstm.mat', 'net');
        
    elseif strcmp(nn_type, 'autoencoder')

        autoenc1 = trainAutoencoder(p_matrix, hiddensize1, ...
            'L2weightRegularization', 0.001, ...
            'SparsityRegularization', 4, ...
            'SparsityProportion', 0.05, ...
            'DecoderTransferFunction', 'purelin');
        
        features1 = encode(autoenc1, p_matrix);
        
        autoenc2 = trainAutoencoder(features1, hiddensize2, ...
            'L2weightRegularization', 0.001, ...
            'SparsityRegularization', 4, ...
            'SparsityProportion', 0.05, ...
            'DecoderTransferFunction', 'purelin', ...
            'ScaleData', false);
        
        features2 = encode(autoenc2, features1);
        
        softnet = trainSoftmaxLayer(features2, t_matrix, 'LossFunction', 'crossentropy');
        
        net = stack(autoenc1, autoenc2, softnet);
        
        net = train(net, p_matrix, t_matrix);
        
        save('autoenc.mat', 'net');
        
    end