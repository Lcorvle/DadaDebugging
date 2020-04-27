function [flag_bugs, delta, ranking, confidence, best_acc] = greedy_duti_lr_cls(X_train,y_train,X_trust,y_trust, num_class, conf,lam, max_iter, max_depth, search_grid, valid_gt_label, test_gt_label, best_acc, batch, data_id)
%% DUTI implementation with binary RBF kernel logistic ridge regression
% Input:
%   X_train     training data,  n x d matrix
%   X_tilde     trusted items,  m x d matrix
%   y_train     training label, n x 1
%   y_tilde     trusted items,  m
%   conf           confidence vector of trusted items, m x 1 vector
%   lam         ridge coefficient of the learner, positive real number
%   budget      examination budget, real number <=n
% Output:
%   ranking     bug flag ranking for prioritization, n x 1 vector, where 
%               ranking(i) = (iteration number when item i first have delta in [1/2, 1]) + (1 - that delta value)
%		Remarks:
%		1. The last term above is for tie-breaking, though there could still be ties.
%		2. The 'ranking' is not necessarily an integer.  Still, investigate the item with the smallest ranking first.
%   deltas      debugging solution (n x T) with value in [0,1], each column is delta^(t)
%   gammas      the gamma in debugging objective O_gamma(), 1 x T vector


    % Set debugging objective to be the objective for kernel logistic
    % regression.
    if nargin < 10
        search_grid = 30;
    end
    if nargin < 9
        max_depth = 10;
    end
    if nargin < 8
        max_iter = 20;
    end
    debugger = @lr_debug_obj;
    cost_func = @lr_cost_obj;

    threshold = 0.5;% threshold for w values to treat a training point as bug
    [n, d] = size(X_train);% training set size
    m = size(X_trust,1);% trusted set size

    % apapt variables
    label = y_train;
    [valid_count, ~] = size(valid_gt_label);
    [test_count, ~] = size(test_gt_label);
    
    X_train = [ones(n, 1), X_train];
    X_trust = [ones(m, 1), X_trust];
    y_train = full(ind2vec((y_train + 1)', num_class))';
    y_trust = full(ind2vec((y_trust + 1)', num_class))';

    % sneaky way to pass parameters to the learner A within matlab optimization
    global global_theta;
    global step_theta;
    global global_delta;
%     global global_ddelta_dtheta;

    c = num_class;
    d = d + 1;

    global_theta = zeros(c, d);
    step_theta = zeros(c, d);
    global_delta = zeros(n, c);
%     global_ddelta_dtheta = zeros(c * d, n * c);

    % find out the maximum gamma_0 value that results in a nonzero w solution,
    % i.e. \nabla_w at w=0 (i.e. the original dirty training data) and gamma=0.
    gamma0 = 0;
    delta0 = y_train;
    [~,grad] = debugger(delta0, X_train, y_train, X_trust, y_trust, gamma0, conf,lam, batch);
    gamma = 2*n*max(sum(y_train .* grad, 2));    

    % Setting up parameters for fmincon    
    delta = delta0;
    confidence = zeros(n, 1);
    ranking = zeros(n, 1);
    flag_bugs = zeros(n, 1) > threshold;
    for iter=1:(max_iter + 1)
        fprintf('\nIter ---------- %d---------------\n',iter);
        gamma = gamma / 2;
%         if iter == (max_iter + 1)
%             gamma = 0;
%         end
%         gamma = 0;
        [delta, step] = gp_optimizer(@(del)debugger(del,X_train,y_train,X_trust,y_trust,gamma,conf,lam, batch), ...
            @(del)cost_func(del,X_train,y_train,X_trust,y_trust,gamma,conf,lam, batch), ...
            delta, max_depth, search_grid, batch);
        violation = sum(y_train .* (y_train - delta), 2);
        iter_flag_bugs = violation > 0.5;
        newly_flag_bugs = iter_flag_bugs & (~ flag_bugs);
%         if iter == 2
%             confidences = max(delta');
%             changed_flag = sum((delta > 0.5) ~= (y_train > 0.5), 2) > 0;
%             confidence(changed_flag) = confidences(changed_flag);
%         end
            
        ranking(newly_flag_bugs) = iter - violation(newly_flag_bugs);
        flag_bugs = iter_flag_bugs;
        
        
        [~, clean_bug_y] = max(delta(:, :), [], 2);
        % [~, clean_bug_y] = max(delta, [], 2);
        clean_bug_y = clean_bug_y - 1;
        y_debug = label;
        y_debug(flag_bugs) = clean_bug_y(flag_bugs);


        acc = sum(valid_gt_label == y_debug(test_count + 1:test_count + valid_count)) / valid_count;
        fprintf('Valid Accuracy: %d / %d = %f\n', sum(valid_gt_label == y_debug(test_count + 1:test_count + valid_count)), valid_count, acc);
        fprintf('Test Accuracy: %d / %d = %f\n', sum(test_gt_label == y_debug(1:test_count)), test_count, sum(test_gt_label == y_debug(1:test_count)) / test_count);
        if acc > best_acc(data_id)
            best_acc(data_id) = acc;
            best_y_debug = y_debug;
            fprintf('===Valid Accuracy: %d / %d = %f\n', sum(valid_gt_label == best_y_debug(test_count + 1:test_count + valid_count)), valid_count, best_acc(data_id));
            fprintf('===Test Accuracy: %d / %d = %f\n', sum(test_gt_label == best_y_debug(1:test_count)), test_count, sum(test_gt_label == best_y_debug(1:test_count)) / test_count);
            theta0 = global_theta;
            theta = LR_train(X_train, delta, lam, theta0, true, batch);
            % save the batch
            result_filename = sprintf('./batch-%d', data_id);
            
            save(result_filename, 'theta');
        
            % save the result
            result_filename = sprintf('greedy_result-batch-%d.mat', data_id);
            
            save(result_filename,'y_debug', 'delta', 'ranking', 'confidence');
        end
    end
end

function [flag_feasible, step, grad_score] = row_direction_search(grad, delta, d)
    step = -grad;

    % project step
    dim = d;
    non_decreasabe_entry = (delta <= 0);
    fixed_all = false(1, d);    
    while true
        fixed_entry = (~fixed_all) & non_decreasabe_entry & (step < sum(step) / dim);
        step(fixed_entry) = 0;

        fixed_all = fixed_all | fixed_entry;
        n_dim = d - sum(fixed_all);
        if (n_dim == dim) | (n_dim == 1)
            dim = n_dim;
            break
        end
        dim = n_dim;
    end
    if dim == 1
        flag_feasible = false;
        step = zeros(1, d);
        grad_score = 0;
        return
    end
    flag_feasible = true;
    % dim
    mutable = ~fixed_all;
    % step
    % mutable
    step(mutable) = step(mutable) - sum(step) / dim;
    grad_score = -1 * dot(step, grad);
    % step
    out_bounded = step < 0;
    scale = -delta(out_bounded) ./ step(out_bounded);
    % scale
    scale = min(scale(:));
    % scale
    step = step * scale;
end

function [flag_feasible, step, grad_score] = gp_step(grad0, delta0, n, d)

    step = zeros(n, d);
    flag_feasible = zeros(n, 1);
    grad_score = zeros(n, 1);
    for i = 1:n
        [i_flag_feasible, i_maximum_step, i_grad_score] = row_direction_search(grad0(i, :), delta0(i, :), d);
        flag_feasible(i) = i_flag_feasible;
        step(i, :) = i_maximum_step;
        grad_score(i) = i_grad_score;
    end
    flag_feasible = logical(flag_feasible);
end

function thresholds = calc_search_grid_thresholds(grad_scores, search_grid)
    % policy = 'avg_scores';
    % policy = 'avg_nums';
    policy = 'min_ranges';
    % another thought, min_ranges but keep min groups > 10

    if strcmp(policy, 'avg_scores')
        min_score = min(grad_scores);
        max_score = max(grad_scores);
        thresholds = min_score + (max_score - min_score) * [0:search_grid - 1] / search_grid;
        return
    end

    if strcmp(policy, 'min_ranges')
        l = length(grad_scores);
        if l <= search_grid
            thresholds = sort(grad_scores);
            return;
        end
        asending_scores_lower = sort(grad_scores);
        asending_scores_higher = zeros(l, 1);
        asending_scores_higher(1:l-1) = asending_scores_lower(2:l);
        gaps = asending_scores_higher - asending_scores_lower;
        [~, max_gap_idx] = sort(gaps, 'descend');
        max_gap_idx = max_gap_idx(1:search_grid - 1);
        thresholds = (asending_scores_higher(max_gap_idx) + asending_scores_lower(max_gap_idx)) / 2;
        thresholds = sort(thresholds);
        thresholds = [min(grad_scores); thresholds];
        return
    end
end

function [flag_continue, delta, cost] = line_search(func, cost_func, delta0, search_grid, depth, batch)
    flag_continue = false;

    [m, d] = size(delta0);

    global global_theta;
    global step_theta;
    [cost0, grad0] = func(delta0);
    global_theta = step_theta;    
    min_group = 10;

    % line search the best based on current
    % deal with some special cases
    [flag_feasible, maximum_step, grad_score] = gp_step(grad0, delta0, m, d);
    num_feasible = sum(flag_feasible);

    % if num_feasible <= min_group
    if num_feasible <= min_group
        delta_to_test = delta0;
        delta_to_test(flag_feasible, :) = delta_to_test(flag_feasible, :)  + maximum_step(flag_feasible, :);
        cost = cost_func(delta_to_test);
        if cost < cost0
            flag_continue = true;
            delta = delta_to_test;
            %fprintf('Searching depth %d Cost %f --> %f\n', depth, cost0, cost);
            return;
        else
            flag_continue = false;
            delta = delta0;
            cost = cost0;
            %fprintf('Searching depth %d end\n', depth);
            return;
        end
    end

    thresholds = calc_search_grid_thresholds(grad_score(flag_feasible), search_grid);

    last_num_changed = -1;
    best_cost = cost0;
    delta = delta0;
    for i = 1:length(thresholds)
        grad_threshold = thresholds(i);
        flag_to_test = grad_score >= grad_threshold;
        num_to_change = sum(flag_to_test);
        if num_to_change == last_num_changed
            continue;
        end
        last_num_changed = num_to_change;
        delta_to_test = delta0;
        delta_to_test(flag_to_test, :) = delta_to_test(flag_to_test, :)  + maximum_step(flag_to_test, :);
        cost = cost_func(delta_to_test);
        %fprintf('Searching depth %d grid: %d/%d, cost: %f, num changed:%d/%d\n', depth, i, search_grid, cost, num_to_change, num_feasible);
        if cost < best_cost
            best_cost = cost;
            delta = delta_to_test;
        end
        % if num_to_change <= min_group
        %     break;
        % end
    end
    flag_continue = best_cost < cost0;
    cost = best_cost;
    if flag_continue
        %fprintf('Searching depth %d Cost %f --> %f\n', depth, cost0, cost);
    else
        %fprintf('Searching depth %d end\n', depth);
    end

end

function [delta, step] = gp_optimizer(func, cost_func, delta0, max_depth, search_grid, batch) 
    delta = delta0;   
    for i = 1:max_depth
        [flag_continue, next_delta, cost] = line_search(func, cost_func, delta, search_grid, i, batch);
        if flag_continue
            delta = next_delta;
        else
            break;
        end
    end
    step = delta - delta0;
end

function cost = lr_cost_obj(delta, X_train, y_train, X_trust, y_trust, gamma, conf, lam, batch)
    [n, d] = size(X_train);
    m = size(X_trust, 1);
    c = size(delta, 2);

    % find the point for a given delta (delta, theta)
    % theta0 = zeros(c, d);
    global global_theta;
    theta0 = global_theta;
    theta = LR_train(X_train, delta, lam, theta0, true, batch);    

    P = lr_predict_prob(X_train, theta);
    % gradiant part trust
    P_trust = lr_predict_prob(X_trust, theta);
    cost = - mean(sum(y_trust .* log(P_trust), 2) .* conf) + ...
        - mean(sum(delta .* log(P), 2)) + ...
        gamma * mean(sum(y_train - (y_train .* delta), 2));
end

function [cost, grad] = lr_debug_obj(delta, X_train, y_train, X_trust, y_trust, gamma, conf, lam, batch)

    [n, d] = size(X_train);
    m = size(X_trust, 1);
    c = size(delta, 2);

    % find the point for a given delta (delta, theta)
    % theta0 = zeros(c, d);
    global global_theta;
    global step_theta;
    theta0 = global_theta;
    theta = LR_train(X_train, delta, lam, theta0, true, batch);    
    step_theta = theta;

    % get c * d implict functions G
    % compute partial derivates of Gkj w.r.t theta and delta

    dG_dtheta = zeros(c * d, c * d);
    dG_ddelta = zeros(c * d, c * n);
%     cube = 20000;
%     num_part = ceil(c * n / cube);
%     for p=1:num_part
%         if p < num_part
%             data = zeros(c * d, cube);
%             save(sprintf('../data/temp-%d.mat', p),'data');
%         else
%             fprintf('updated\n');
%             if mod(c * n, cube) > 0
%                 data = zeros(c * d, mod(c * n, cube));
%                 save(sprintf('../data/temp-%d.mat', p),'data');
%             else
%                 data = zeros(c * d, cube);
%                 save(sprintf('../data/temp-%d.mat', p),'data');
%             end
%         end
%     end
    P = lr_predict_prob(X_train, theta);

    ind_row = 1;
    for k = 1:c

        % make k col Y
        k_col_Y = zeros(1, c);
        k_col_Y(k) = 1;
        k_col_Y = ones(n, 1) * k_col_Y;

        % make k col P
        k_col_P = P(:, k);

        % make common
        common_mul = 1 / n * bsxfun(@times, k_col_Y - P, k_col_P);
        for j = 1:d            
            dgelement_dtheta = bsxfun(@times, common_mul, X_train(:, j))' * X_train;
            dgelement_dtheta(k, j) = dgelement_dtheta(k, j) + lam;
            dG_dtheta(ind_row, :) = dgelement_dtheta(:);

            dgelement_ddelta = zeros(n, c);
            dgelement_ddelta(:, k) = -1 / n * X_train(:, j);
            dG_ddelta(ind_row,:) = dgelement_ddelta(:);
%             num_part = ceil(c * n / cube);
%             for p=1:num_part
%                 if p < num_part
%                     data = load(sprintf('../data/temp-%d.mat', p));
%                     data = data.data;
%                     data(ind_row,:) = dgelement_ddelta(cube*(p-1) + 1:cube*p);
%                     save(sprintf('../data/temp-%d.mat', p),'data');
%                 else
%                     data = load(sprintf('../data/temp-%d.mat', p));
%                     data = data.data;
%                     data(ind_row,:) = dgelement_ddelta(cube*(p-1) + 1:c*n);
%                     save(sprintf('../data/temp-%d.mat', p),'data');
%                 end
%             end

            ind_row = ind_row + 1;
        end
    end
    dtheta_ddelta = - pinv(dG_dtheta) * dG_ddelta;
%     temp = - pinv(dG_dtheta);
%     num_part = ceil(c * n / cube);
%     for p=1:num_part
%         data = load(sprintf('../data/temp-%d.mat', p));
%         data = data.data;
%         data = temp * data;
%         data = data';
%         save(sprintf('../data/temp-%d.mat', p),'data');
%     end
    

    % gradiant part trust
    P_trust = lr_predict_prob(X_trust, theta);
    nbla_ltrust_theta = -1 / m * (y_trust - P_trust)' * bsxfun(@times, X_trust, conf);
    dltrust_ddelta = reshape(dtheta_ddelta' * nbla_ltrust_theta(:), [n c]);
%     dltrust_ddelta = zeros(c * n, 1);
%     num_part = ceil(c * n / cube);
%     for p=1:num_part
%         if p < num_part
%             data = load(sprintf('../data/temp-%d.mat', p));
%             data = data.data * nbla_ltrust_theta(:);
%             dltrust_ddelta(cube*(p-1) + 1:cube*p) = data(:);
%         else
%             data = load(sprintf('../data/temp-%d.mat', p));
%             data = data.data * nbla_ltrust_theta(:);
%             dltrust_ddelta(cube*(p-1) + 1:c*n) = data(:);
%         end
%     end
%     dltrust_ddelta = reshape(dltrust_ddelta, [n, c]);
    
    

    % gradiant part noisy
    nabla_lnoisy_theta = -1 / n * (delta - P)' * X_train;
    dlnoisy_ddelta = reshape(dtheta_ddelta' * nabla_lnoisy_theta(:), [n c]) -1 / n * log(P);

    % gradiant part distance
    ddist_ddelta = - gamma / n * y_train;

    cost = - mean(sum(y_trust .* log(P_trust), 2) .* conf) + ...
        - mean(sum(delta .* log(P), 2)) + ...
        gamma * mean(sum(y_train - (y_train .* delta), 2));
    grad = dltrust_ddelta + dlnoisy_ddelta + ddist_ddelta;
end

function h = lr_predict_prob(X, theta)
    h = exp(X * theta');
    sum_h = sum(h, 2);
    h = bsxfun(@rdivide, h, sum_h);
end


