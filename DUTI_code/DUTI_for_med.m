close all;clear all;

part = 1;
greedy = true;

% load data
data = load('../data/binary_naogeng_pca64.mat');
X_trust = double(data.trusted_features);
trust_item_label = data.trusted_gt_labels;
ids = data.ids;
trust_item_label = double(trust_item_label');
feat = double(data.features);
label = double(data.labels);
label = double(label');
gt_label = double(data.gt_gt_labels);
gt_label = double(gt_label');
gt_index = data.gt_indexes;
gt_index = int64(gt_index') + 1;
% process trust item
[num_data, dim_data] = size(feat);
num_data = int64(num_data);
[num_trust, ~] = size(X_trust);
trust_conf = ones(num_trust, 1);
[num_gt, ~] = size(gt_index);

clear data;

% debug using DUTI -----------------------------------------------------
lam = 1; 	% L2 regularization weight of learner
c_value = 100;  % Confidence parameters on trusted items are set to 100.
%conf = c_value * ones(num_trust, 1);
conf = trust_conf * c_value;
max_depth = 20;
search_grid = 50;
num_class = 2;
max_iter = 3;
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
[~, clean_bug_y] = max(delta(bugs, :), [], 2);
% [~, clean_bug_y] = max(delta, [], 2);
clean_bug_y = clean_bug_y - 1;
% bugs = (rankings < 4) & (rankings > 0);

y_debug(bugs) = clean_bug_y;

num_bugs = sum(bugs);
fprintf('number of bug found %d\n', num_bugs);
fprintf('Accuracy: %d / %d = %f\n', sum(label(gt_index) == gt_label), num_gt, sum(label(gt_index) == gt_label) / num_gt);
fprintf('Accuracy: %d / %d = %f\n', sum(y_debug(gt_index) == gt_label), num_gt, sum(y_debug(gt_index) == gt_label) / num_gt);

% save the result
if greedy
    result_filename = sprintf('greedy_result-%d-finetune-for_vis_201909091410_pca_128.mat', part);
else
    result_filename = sprintf('result-%d-finetune2.mat', part);
end
% confidence_ids = ids(confidence > 0.9, :);
save(result_filename,'y_debug', 'delta', 'rankings', 'confidence');


