close all;clear all;

part = 1;
greedy = true;

data_mode = 'simulate';
if strcmp(data_mode, 'real')
    % load data
    data = load('../data/cifar.mat');
    trust_conf = double(data.trusted_confidence');
    X_trust = double(data.trusted_feature);
    trust_item_label = data.trusted_label;
    ids = data.ids;
    trust_item_label = double(trust_item_label');
    feat = double(data.feature);
    label = double(data.label);
    label = double(label');
    gt_label = double(data.gt_label);
    gt_label = double(gt_label');
    % process trust item
    [num_data, dim_data] = size(feat);
    num_data = int64(num_data);
    [num_trust, ~] = size(X_trust);
else
    if strcmp(data_mode, 'simulate')
        % simulated data for debug
        data = load('dataset.mat');
        trust_item_index = [1, 4];
        trust_item_label = [3, 2];
        trust_item_label = trust_item_label';
        feat = double(data.feature);
        feat = feat(1:5, :);
        label = [1, 1, 2, 3, 3];
        label = label';
    else
        % load data
        data = load('dataset.mat');
        trust_item = load('trust_item.mat');
        trust_item_index = trust_item.indexs;
        trust_item_index = trust_item_index + 1;
        trust_item_index = trust_item_index(:);
        trust_item_label = trust_item.label;
        trust_item_label = double(trust_item_label');
        trust_item_label = trust_item_label(:);
        feat = double(data.feature);
        feat = feat(:, :);
        label = double(data.label);
        label = double(label');
        label = label(:);
    end
    % process trust item
    [num_data, dim_data] = size(feat);
    num_data = int64(num_data);
    [temp ,num_trust] = size(trust_item_index);
    X_trust = feat(trust_item_index, :);
    train_label_trust_item = label(trust_item_index);
    fprintf('number of train points %d, num bugs in trusted items %d\n', num_data, sum(train_label_trust_item ~= trust_item_label));
end

% [N, D] = size(gt_label);

clear data;
clear trust_item;


% debug using DUTI -----------------------------------------------------
lam = 1; 	% L2 regularization weight of learner
c_value = 100;  % Confidence parameters on trusted items are set to 100.
conf = c_value * ones(num_trust, 1);
% conf = trust_conf * c_value;
max_depth = 20;
search_grid = 50;
num_class = 4;
max_iter = 1;
tic
if greedy
    [bugs, delta, rankings, confidence] = greedy_duti_lr_cls(feat, label, X_trust, trust_item_label, ...
        num_class, conf, lam, max_iter, max_depth, search_grid);
else
    [bugs, delta, rankings] = duti_lr_cls(feat, label, X_trust, trust_item_label, ...
        num_class, conf, lam, max_iter);
end
toc

if greedy
    fprintf('greedy_DUTI done\n');
else
    fprintf('DUTI done\n');
end


y_debug = label;
% [~, clean_bug_y] = max(delta(bugs, :), [], 2);
[~, clean_bug_y] = max(delta, [], 2);
clean_bug_y = clean_bug_y - 1;
% bugs = (rankings < 4) & (rankings > 0);

y_debug(bugs) = clean_bug_y(bugs);

num_bugs = sum(bugs);
fprintf('number of bug found %d\n', num_bugs);
fprintf('Accuracy: %d / %d = %f\n', sum(label == gt_label), N, sum(label == gt_label) / N);
fprintf('Accuracy: %d / %d = %f\n', sum(y_debug == gt_label), N, sum(y_debug == gt_label) / N);

% save the result
if greedy
    result_filename = sprintf('greedy_result-%d-finetune-for_vis_201909091410_pca_128.mat', part);
else
    result_filename = sprintf('result-%d-finetune2.mat', part);
end
% confidence_ids = ids(confidence > 0.9, :);
save(result_filename,'y_debug', 'delta', 'rankings', 'confidence');


