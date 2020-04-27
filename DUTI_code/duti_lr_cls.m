function [flag_bugs, delta, ranking] = duti_lr_cls(X_train,y_train,X_trust,y_trust, num_class, conf,lam, max_iter)
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
%   flag_bugs   the flag of bug


    % Set debugging objective to be the objective for kernel logistic
    % regression.
    if nargin < 8
        max_iter = 20;
    end
    debugger = @lr_debug_obj;

    threshold = 0.5;% threshold for w values to treat a training point as bug
    [n, d] = size(X_train);% training set size
    m = size(X_trust,1);% trusted set size

    % apapt variables
    X_train = [ones(n, 1), X_train];
    X_trust = [ones(m, 1), X_trust];
    y_train = full(ind2vec((y_train + 1)', num_class))';
    y_trust = full(ind2vec((y_trust + 1)', num_class))';

    % sneaky way to pass parameters to the learner A within matlab optimization
    global global_theta;
    global step_theta;
    global global_delta;
    global global_ddelta_dtheta;

    c = num_class;

    d = d + 1;

    global_theta = zeros(c, d);
    step_theta = zeros(c, d);
    global_delta = zeros(n, c);
    global_ddelta_dtheta = zeros(c * d, n * c);

    % find out the maximum gamma_0 value that results in a nonzero w solution,
    % i.e. \nabla_w at w=0 (i.e. the original dirty training data) and gamma=0.
    gamma0 = 0;
    delta0 = y_train;
    [~,grad] = debugger(delta0, X_train, y_train, X_trust, y_trust, gamma0, conf,lam);
    gamma = 2*n*max(sum(y_train .* grad, 2));    

    % Setting up parameters for fmincon    
    delta = delta0;
    ranking = zeros(n, 1);
    flag_bugs = zeros(n, 1) > threshold;
    for iter=1:(max_iter + 1)
        fprintf('\nIter ---------- %d---------------\n',iter);
        gamma = gamma / 2;
        if iter == (max_iter + 1)
            gamma = 0;
        end
        gamma
        [delta, step] = gp_optimizer(@(del)debugger(del,X_train,y_train,X_trust,y_trust,gamma,conf,lam), ...
            delta);   
        violation = sum(y_train .* (y_train - delta), 2);
        iter_flag_bugs = violation > 0.5;
        newly_flag_bugs = iter_flag_bugs & (~ flag_bugs);
        ranking(newly_flag_bugs) = iter - violation(newly_flag_bugs);
        flag_bugs = iter_flag_bugs;
    end
    
end

function [step] = gp_step(grad0, delta0, n, d, lr)
    step = zeros(n, d);
    for i = 1:n
        step(i, :) = row_direction_search(grad0(i, :), delta0(i, :), d, lr);
    end
end

function [delta, step] = gp_optimizer(func, delta0)
    [m, d] = size(delta0);

    global global_theta;
    global step_theta;

    [cost0, grad0] = func(delta0);
    global_theta = step_theta;    
    fprintf('Cost %f\n', cost0);        

    lr = 1000;
    lr_to_tol = 1e-5;
    func_to_tol = -1e-10;

    step = gp_step(grad0, delta0, m, d, lr);
    last_max_step = max(max(abs(step))); 

    while last_max_step < 0.1
        lr = lr * 10;
        step = gp_step(grad0, delta0, m, d, lr);
        max_step = max(max(abs(step)));

        if max_step <= last_max_step
            lr = lr * 0.1;
            break;
        end
        last_max_step = max_step;
    end
    
    while true
        step = gp_step(grad0, delta0, m, d, lr);
        delta = delta0 + step;
        % delta(delta < 0) = 0;
        [cost, grad] = func(delta);
        if cost - cost0 > func_to_tol        
            % fprintf('lr  %f not feasible %f\n',lr, cost);    
            lr = lr * 0.1;
            if lr < lr_to_tol
                break;
            end
            continue;            
        end
        cost0 = cost;
        grad0 = grad;
        delta0 = delta;
        global_theta = step_theta;
        % lr = 0.1;
        fprintf('lr %f step %f Cost %f\n', lr, max(max(abs(step))), cost0);
    end
end

function step = row_direction_search(grad, delta, d, lr)
    step = -lr * grad;

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
        step = zeros(1, d);
        return
    end
    % dim
    mutable = ~fixed_all;
    % step
    % mutable
    step(mutable) = step(mutable) - sum(step) / dim;
    % step
    out_bounded = step < 0;
    scale = -delta(out_bounded) ./ step(out_bounded);
    % scale
    scale = min(1, min(scale(:)));
    % scale
    step = step * scale;
end

function [cost, grad] = lr_debug_obj(delta, X_train, y_train, X_trust, y_trust, gamma, conf, lam)

    [n, d] = size(X_train);
    m = size(X_trust, 1);
    c = size(delta, 2);

    % find the point for a given delta (delta, theta)
    % theta0 = zeros(c, d);
    global global_theta;
    global step_theta;
    theta0 = global_theta;
    theta = LR_train(X_train, delta, lam, theta0, true);    
    step_theta = theta;

    % get c * d implict functions G
    % compute partial derivates of Gkj w.r.t theta and delta

    dG_dtheta = zeros(c * d, c * d);
    dG_ddelta = zeros(c * d, c * n);

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
            dG_ddelta(ind_row, :) = dgelement_ddelta(:);

            ind_row = ind_row + 1;
        end
    end

    dtheta_ddelta = - pinv(dG_dtheta) * dG_ddelta;

    % gradiant part trust
    P_trust = lr_predict_prob(X_trust, theta);
    nbla_ltrust_theta = -1 / m * (y_trust - P_trust)' * bsxfun(@times, X_trust, conf);
    dltrust_ddelta = reshape(dtheta_ddelta' * nbla_ltrust_theta(:), [n c]);

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


