close all;clear all;

part = 1;
greedy = true;

% load data
data = load('../data/game_trusted2_3class_plan_D.mat');
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
valid_ids = data.valid_ids;
valid_feat = double(data.valid_features);
valid_label = double(data.valid_noisy_labels);
valid_label = double(valid_label');
valid_gt_label = double(data.valid_trusted_labels);
valid_gt_label = double(valid_gt_label');
[valid_count, ~] = size(valid_label);
[test_count, ~] = size(test_label);
feat = cat(1, test_feat, cat(1, valid_feat, feat));
ids = cat(1, test_ids, cat(1, valid_ids, ids));
label = cat(1, test_label, cat(1, valid_label, label));
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


best_acc = sum(valid_gt_label == label(test_count + 1:test_count + valid_count)) / valid_count;
best_y_debug = label;
best_ranking = 0;
best_c_value = 0;

search_range = [0.01, 0.02, 0.05, 0.1, 0.15, 0.2];

for search_iter=1:7
    c_value = search_range(search_iter);
    % debug using DUTI -----------------------------------------------------
    lam = 1; 	% L2 regularization weight of learner
    conf = trust_conf * c_value;
    max_depth = 20;
    search_grid = 30;
    num_class = 3;
    max_iter = 8;
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
    for i=min(rankings):max(rankings)+1
        bugs = (rankings < i);
        y_debug = label;
        y_debug(bugs) = clean_bug_y(bugs);

        num_bugs = sum(bugs);
        acc = sum(valid_gt_label == y_debug(test_count + 1:test_count + valid_count)) / valid_count;
        fprintf('ranking<%d===============\nnumber of bug found %d\n', i, num_bugs);
        fprintf('Valid Accuracy: %d / %d = %f\n', sum(valid_gt_label == y_debug(test_count + 1:test_count + valid_count)), valid_count, acc);
        fprintf('Test Accuracy: %d / %d = %f\n', sum(test_gt_label == y_debug(1:test_count)), test_count, sum(test_gt_label == y_debug(1:test_count)) / test_count);
        if acc > best_acc
            best_acc = acc;
            best_y_debug = y_debug;
            best_ranking = i;
            best_c_value = c_value;
            % save the result
            if greedy
                result_filename = sprintf('greedy_result_%d.mat', part);
            else
                result_filename = sprintf('result_%d.mat', part);
            end
            save(result_filename,'y_debug', 'delta', 'rankings', 'confidence', 'ids');
        end
    end
end
fprintf('Best accuracy: best_ranking=%d, best_c_value=%d\n', best_ranking, best_c_value);
fprintf('Valid Accuracy: %d / %d = %f\n', sum(valid_gt_label == best_y_debug(test_count + 1:test_count + valid_count)), valid_count, best_acc);
fprintf('Test Accuracy: %d / %d = %f\n', sum(test_gt_label == best_y_debug(1:test_count)), test_count, sum(test_gt_label == best_y_debug(1:test_count)) / test_count);        
    