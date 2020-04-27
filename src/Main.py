"""
Main script to debugging data.
Created by shouxing, 2019/6/27
"""
from src import DataLoader, ResultSaver
from src.DUTI import DUTI
from src.GreedyDUTI import GreedyDUTI
import numpy as np
import os


def run_duti(dataset_path, trusted_item_path, feature_dir, result_path):
    """
    Run the duti algorithm to correct the label of training items.
    :param dataset_path: the path of the training dataset.
    :param trusted_item_path: the path of the trusted items.
    :param result_path: the path to save the result.
    :return: this function return nothing.
    """
    # load data from a specified txt file.
    print('Loading dataset...')
    flag, dataset = DataLoader.load_from_txt(dataset_path)
    if flag != 'Train':
        print('warning: the ' + dataset_path + ' is not a train file.')
    print('Number of training data:', len(dataset))
    print('Loading trusted items...')
    flag, trusted_items = DataLoader.load_from_txt(trusted_item_path)
    if flag != 'Trusted':
        print('warning: the ' + trusted_item_path + ' is not a trusted file.')
    print('Number of trusted items:', len(trusted_items))
    print('Merging dataset and trusted items...')
    data, labelnames, flag = DataLoader.merge_dataset_and_trusted_items(dataset, trusted_items)

    # use duti to fix each dataset
    if flag:
        num_class = len(labelnames)
        duti = GreedyDUTI(num_class=num_class, max_iter=20)

        # init input for the duti algorithm
        print('Loading features for training data...')
        feature = [DataLoader.get_feature(os.path.join(feature_dir, id + '.npy')) for id in data['ids']]
        print('Loading features for trusted items...')
        trusted_feature = [DataLoader.get_feature(os.path.join(feature_dir, id + '.npy')) for id in data['trusted_ids']]
        confidence = np.ones_like(data['trusted_labels'])
        np.savez(dataset_path.replace('Train', 'Compressed').replace('.txt', '.npz'), feature=np.array(feature),
                 label=np.array(data['labels']), trusted_feature=np.array(trusted_feature),
                 trusted_label=np.array(data['trusted_labels']))
        print('Running the duti algorithm...')
        bugs, delta, rankings = duti.fit_transform(np.array(feature), np.array(data['labels']),
                                                   np.array(trusted_feature), np.array(data['trusted_labels']),
                                                   confidence)
        y_debug = np.array(data['labels'], copy=True)
        clean_bug_y = np.argmax(delta[bugs, :], axis=1)
        y_debug[bugs] = clean_bug_y
        np.savez(result_path.replace('Result', 'Compressed').replace('.txt', '.npz'), y_debug=np.array(y_debug),
                 delta=np.array(delta), rankings=np.array(rankings))

        result = []
        for i, id in enumerate(data['ids']):
            if id in data['trusted_ids']:
                result.append({
                    'id': id,
                    'frame': data['frames'][i],
                    'label_name': labelnames[data['trusted_labels'][i]],
                    'flag': 0,
                    'other': data['others'][i]
                })
            else:
                result.append({
                    'id': id,
                    'frame': data['frames'][i],
                    'label_name': labelnames[y_debug[i]],
                    'flag': 0,
                    'other': data['others'][i]
                })

        print('Saving the result...')
        ResultSaver.save_as_txt(result_path, result)
    else:
        num_class = 2
        duti = GreedyDUTI(num_class=num_class, max_iter=20, method='decision tree')
        result = []
        for index, dataset in enumerate(data):
            # init input for the duti algorithm
            print('Loading features for training data...')
            feature = [DataLoader.get_feature(os.path.join(feature_dir, id + '.npy')) for id in dataset['ids']]
            print('Loading features for trusted items...')
            trusted_feature = [DataLoader.get_feature(os.path.join(feature_dir, id + '.npy')) for id in dataset['trusted_ids']]
            confidence = np.ones_like(dataset['trusted_labels'])
            np.savez(dataset_path.replace('Train', 'Compressed').replace('.txt', str(index) + '.npz'),
                     feature=np.array(feature),
                     label=np.array(dataset['labels']), trusted_feature=np.array(trusted_feature),
                     trusted_label=np.array(dataset['trusted_labels']))
            print('Running the duti algorithm...')
            bugs, delta, rankings = duti.fit_transform(np.array(feature), np.array(dataset['labels']),
                                                       np.array(trusted_feature), np.array(dataset['trusted_labels']),
                                                       confidence)

            print('Correct', bugs.sum(), 'bugs')

            y_debug = np.array(dataset['labels'], copy=True)
            clean_bug_y = np.argmax(delta[bugs, :], axis=1)
            y_debug[bugs] = clean_bug_y
            np.savez(result_path.replace('Result', 'Compressed').replace('.txt', str(index) + '.npz'), y_debug=np.array(y_debug),
                     delta=np.array(delta), rankings=np.array(rankings))

            # for bug in range(len(bugs)):
            #     if bugs[bug]:
            #         print(dataset['ids'][bug], dataset['labels'][bug], 'to', y_debug[bug])
            for i, id in enumerate(dataset['ids']):
                if id in dataset['trusted_ids']:
                    result.append({
                        'id': id,
                        'frame': dataset['frames'][i],
                        'label_name': labelnames[index],
                        'flag': dataset['trusted_labels'][dataset['trusted_ids'].index(id)] + 1,
                        'other': dataset['others'][i]
                    })
                else:
                    result.append({
                        'id': id,
                        'frame': dataset['frames'][i],
                        'label_name': labelnames[index],
                        'flag': y_debug[i] + 1,
                        'other': dataset['others'][i]
                    })
        print('Saving the result...')
        ResultSaver.save_as_txt(result_path, result)
    print('===finish correcting the dataset:', dataset_path,
          'by', trusted_item_path,
          'and save the result in', result_path, '===')


def food_duti(path):
    data = np.load(path)
    feature = data['features']
    label = data['labels']
    trusted_feature = data['trusted_features']
    trusted_label = data['trusted_labels']
    gt_indexes = data['gt_indexes']
    gt_gt_labels = data['gt_gt_labels']
    print('comment: accuracy before duti:', sum(label[gt_indexes] == gt_gt_labels), '/', len(gt_indexes), '=', sum(label[gt_indexes] == gt_gt_labels) / len(gt_indexes))
    confidence = np.ones_like(label)
    duti = GreedyDUTI(num_class=54, max_iter=10)
    bugs, delta, rankings = duti.fit_transform(feature, label, trusted_feature, trusted_label, confidence)

    print('comment: Correct', bugs.sum(), 'bugs')

    y_debug = np.array(label, copy=True)
    clean_bug_y = np.argmax(delta[bugs, :], axis=1)
    y_debug[bugs] = clean_bug_y
    np.savez('result_of_food_comment_plain1.npz', bugs=bugs, delta=delta, rankings=rankings)
    print('comment: accuracy after duti:', sum(y_debug[gt_indexes] == gt_gt_labels), '/', len(gt_indexes), '=', sum(y_debug[gt_indexes] == gt_gt_labels) / len(gt_indexes))
    ##############################################

    label = data['prediction_labels']
    print('prediction: accuracy before duti:', sum(label[gt_indexes] == gt_gt_labels), '/', len(gt_indexes), '=',
          sum(label[gt_indexes] == gt_gt_labels) / len(gt_indexes))
    duti = GreedyDUTI(num_class=54, max_iter=20)
    bugs, delta, rankings = duti.fit_transform(feature, label, trusted_feature, trusted_label, confidence)

    print('prediction: Correct', bugs.sum(), 'bugs')

    y_debug = np.array(label, copy=True)
    clean_bug_y = np.argmax(delta[bugs, :], axis=1)
    y_debug[bugs] = clean_bug_y
    np.savez('result_of_food_prediction_plain1.npz', bugs=bugs, delta=delta, rankings=rankings)
    print('prediction: accuracy after duti:', sum(y_debug[gt_indexes] == gt_gt_labels), '/', len(gt_indexes), '=',
          sum(y_debug[gt_indexes] == gt_gt_labels) / len(gt_indexes))


if __name__ == '__main__':
    # run_duti("../data/Train_2522.txt", "../data/Trusted_2522.txt", "../data/game_features", "../result/Result_2522_method=dt_min_leaf=2_min_split=5.txt")
    # run_duti("../data/Train_2606.txt", "../data/Trusted_2606.txt", "../data/game_feature", "../result/Result_2606.txt")
    food_duti('../data/food_plain_pca128.npz')