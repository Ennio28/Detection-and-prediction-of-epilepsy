    function T_matrix = indexing_target(T)
    
    ictal_idx = find(T == 1); % Calculating the indices of non-zero values that rappresent ictal-class
    ictal_idx_length = length(ictal_idx);   
    
    for i = 1 : ictal_idx_length
        if (i == 1)
            T(ictal_idx(i) - 900 : ictal_idx(i) - 1) = 2;
        else
            before_idx = ictal_idx(i-1);
            cur_idx = ictal_idx(i);
            % cur_idx is the beginning of the next
            % seizure and the before_idx the end of the previous seizure
            if cur_idx - before_idx > 1 
                T(cur_idx - 900 : cur_idx - 1) = 2; % pre-ictal
                T(before_idx + 1 : before_idx + 301) = 1; % post-ictal
            end
        end
    end
    
    T(cur_idx + 1 : cur_idx + 301) = 1; % last seizure post-ictal
    
    T(T == 1) = 3; % ictal from 1 to 3
    T(T == 0) = 1; % interictal from 0 to 1
    
    % Create a matrix of three columns
    T_matrix = zeros(length(T), 3);
    for i = 1 : length(T)
        T_matrix(i, T(i)) = 1;
    end

end

 
  