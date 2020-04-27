close all;clear all;

part = 1;
greedy = true;

% load data
data = load('../data/food_plain10_pca128_prediction_only_groundtruth.mat');
X_trust = double(data.trusted_features);
trust_item_label = data.trusted_labels;
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
c_value = 1;  % Confidence parameters on trusted items are set to 100.
%conf = c_value * ones(num_trust, 1);
conf = trust_conf * c_value;
max_depth = 20;
search_grid = 30;
num_class = 10;
max_iter = 10;
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
[~, clean_bug_y] = max(delta(:, :), [], 2);
% [~, clean_bug_y] = max(delta, [], 2);
clean_bug_y = clean_bug_y - 1;
best_acc = 0;
for i=2:max(rankings)+1
    bugs = (rankings < i) & (rankings > 0);

    y_debug(bugs) = clean_bug_y(bugs);

    num_bugs = sum(bugs);
    fprintf('ranking<%d===============\nnumber of bug found %d\n', i, num_bugs);
    fprintf('Accuracy: %d / %d = %f\n', sum(label(gt_index) == gt_label), num_gt, sum(label(gt_index) == gt_label) / num_gt);
    fprintf('Accuracy: %d / %d = %f\n', sum(y_debug(gt_index) == gt_label), num_gt, sum(y_debug(gt_index) == gt_label) / num_gt);
    acc = sum(y_debug(gt_index) == gt_label) / num_gt;
    if acc > best_acc
        best_acc = acc;
        % save the result
        if greedy
            result_filename = sprintf('greedy_food_plain10_pca128_prediction_only_groundtruth-iter%d.mat', part);
        else
            result_filename = sprintf('food_plain10_pca128_prediction_only_groundtruth-iter%d.mat', part);
        end
        % confidence_ids = ids(confidence > 0.9, :);
        save(result_filename,'y_debug', 'delta', 'rankings', 'confidence', 'ids');
    end
end
fprintf('Best accuracy: %f\n', best_acc);
    
    