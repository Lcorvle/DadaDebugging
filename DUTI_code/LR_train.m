function alpha = LR_train(X, y, lam, alpha0, preprocessed, batch)
    if nargin == 4
        preprocessed = false;
    end

    options = optimoptions('fminunc','Display','off','Algorithm','quasi-newton',...
        'SpecifyObjectiveGradient',true,'FunctionTolerance',1e-6,...
        'MaxIterations',4000,'StepTolerance',1e-6,'UseParallel',true);

    [n, d] = size(X);
    [num_class, ~] = size(alpha0);
    if ~ preprocessed
        X = [ones(n, 1) X];
        y = full(ind2vec((y + 1)', num_class))';
    end

    alpha = fminunc(@(alpha)costLR(alpha, X, y, lam, batch),alpha0,options);
end

function [J, grad] = costLR(alpha, X, y, lam, batch)
    [n, ~] = size(X);
    delta_batch = 0;
    if batch ~= false
        sum_batch = sum(alpha - batch);
    end
    h = exp(X * alpha');
    sum_h = sum(h, 2);
    h = bsxfun(@rdivide, h, sum_h);

    J_err = -mean(sum(y .* log(h), 2));
    grad_err = -1 / n * (y - h)' * X;

    J_stream = 0.01 * sumsqr(alpha - batch);
    grad_stream = 0.02 * (alpha - batch);
    
    J_reg = lam / 2 * sumsqr(alpha);
    grad_reg = lam * alpha;
    J = J_err + J_reg + J_stream;
    %fprintf('ERR: %f REG: %f J: %f\n', J_err, J_reg, J);
    grad = grad_err + grad_reg + grad_stream;
end