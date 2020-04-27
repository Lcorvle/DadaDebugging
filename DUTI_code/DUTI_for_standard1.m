close all;clear all;

part = 1;
greedy = true;

% load data
data = load('../data/game_trusted2_3class_plan_C.mat');
X_trust = double(data.train_features);
trust_item_label = data.train_trusted_labels;
trust_item_label = double(trust_item_label');
trust_ids = data.train_ids;
ids = data.ids;
feat = double(data.features);
label = double(data.labels);
label = double(label');
test_ids = data.test_ids;
test_feat = double(data.test_features);
test_label = double(data.test_noisy_labels);
test_label = double(test_label');
test_gt_label = double(data.test_trusted_labels);
test_gt_label = double(test_gt_label');
valid1_ids = data.valid1_ids;
valid1_feat = double(data.valid1_features);
valid1_label = double(data.valid1_noisy_labels);
valid1_label = double(valid1_label');
valid1_gt_label = double(data.valid1_trusted_labels);
valid1_gt_label = double(valid1_gt_label');
[valid1_count, ~] = size(valid1_label);
valid2_ids = data.valid2_ids;
valid2_feat = double(data.valid2_features);
valid2_label = double(data.valid2_noisy_labels);
valid2_label = double(valid2_label');
valid2_gt_label = double(data.valid2_trusted_labels);
valid2_gt_label = double(valid2_gt_label');
[valid2_count, ~] = size(valid2_label);
[test_count, ~] = size(test_label);
feat = cat(1, test_feat, cat(1, valid1_feat, cat(1, valid2_feat, feat)));
ids = cat(1, test_ids, cat(1, valid1_ids, cat(1, valid2_ids, ids)));
label = cat(1, test_label, cat(1, valid1_label, cat(1, valid2_label, label)));
% 
% feat = feat(1:100,:);
% ids = ids(1:100,:);
% label = label(1:100,:);
% X_trust = X_trust(1:100,:);
% trust_item_label = trust_item_label(1:100,:);
% trust_ids = trust_ids(1:100,:);

% process trust item
[num_data, dim_data] = size(feat);
num_data = int64(num_data);
[num_trust, ~] = size(X_trust);
trust_conf = ones(num_trust, 1);

clear data;
% debug using DUTI -----------------------------------------------------
lam = 1; 	% L2 regularization weight of learner
c_value = 1;  % Confidence parameters on trusted items are set to 100.
%conf = c_value * ones(num_trust, 1);
conf = trust_conf * c_value;
max_depth = 20;
search_grid = 30;
num_class = 3;
max_iter = 20;
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

[~, clean_bug_y] = max(delta(:, :), [], 2);
% [~, clean_bug_y] = max(delta, [], 2);
clean_bug_y = clean_bug_y - 1;

best_acc1 = sum(valid1_gt_label == label(test_count + 1:test_count + valid1_count)) / valid1_count;
best_acc2 = sum(valid2_gt_label == label(test_count + valid1_count + 1:test_count + valid1_count + valid2_count)) / valid2_count;
best_y_debug1 = label;
best_y_debug2 = label;
for i=min(rankings):max(rankings)+1
    bugs = (rankings < i);
    y_debug = label;
    y_debug(bugs) = clean_bug_y(bugs);

    num_bugs = sum(bugs);
    acc1 = sum(valid1_gt_label == y_debug(test_count + 1:test_count + valid1_count)) / valid1_count;
    acc2 = sum(valid2_gt_label == y_debug(test_count + valid1_count + 1:test_count + valid1_count + valid2_count)) / valid2_count;
    fprintf('ranking<%d===============\nnumber of bug found %d\n', i, num_bugs);
    fprintf('Valid1 Accuracy: %d / %d = %f\n', sum(valid1_gt_label == y_debug(test_count + 1:test_count + valid1_count)), valid1_count, acc1);
    fprintf('Valid2 Accuracy: %d / %d = %f\n', sum(valid2_gt_label == y_debug(test_count + valid1_count + 1:test_count + valid1_count + valid2_count)), valid2_count, acc2);
    fprintf('Test Accuracy: %d / %d = %f\n', sum(test_gt_label == y_debug(1:test_count)), test_count, sum(test_gt_label == y_debug(1:test_count)) / test_count);
    if acc1 > best_acc1
        best_acc1 = acc1;
        best_y_debug1 = y_debug;
    end
    if acc2 > best_acc2
        best_acc2 = acc2;
        best_y_debug2 = y_debug;
    end
end
fprintf('Best accuracy:\n');
fprintf('Valid1 Accuracy: %d / %d = %f\n', sum(valid1_gt_label == best_y_debug1(test_count + 1:test_count + valid1_count)), valid1_count, best_acc1);
fprintf('Test Accuracy: %d / %d = %f\n', sum(test_gt_label == best_y_debug1(1:test_count)), test_count, sum(test_gt_label == best_y_debug1(1:test_count)) / test_count);
fprintf('Valid2 Accuracy: %d / %d = %f\n', sum(valid2_gt_label == best_y_debug2(test_count + valid1_count + 1:test_count + valid1_count + valid2_count)), valid2_count, best_acc2);
fprintf('Test Accuracy: %d / %d = %f\n', sum(test_gt_label == best_y_debug2(1:test_count)), test_count, sum(test_gt_label == best_y_debug2(1:test_count)) / test_count);
    
    
    