from scipy import io
import numpy as np
from src import DataLoader
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN, KMeans
from sklearn import metrics
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA



def transform_npz_to_mat(npz_path, mat_path):
    print('=' * 20)
    mat = np.load(npz_path)
    res = {}
    for key in mat.keys():
        res[key] = mat[key]
        print(key, mat[key].shape)
    io.savemat(mat_path, res)


def transform_mat_to_npz(mat_path, npz_path):
    mat = io.loadmat(mat_path)
    np.savez(npz_path,
             y_debug=np.array(mat['y_debug']),
             confidence=np.array(mat['confidence']),
             delta=np.array(mat['delta']), rankings=np.array(mat['rankings']))


def check_which_be_correct(training_file, result_file, figure_path):
    import shutil, os
    f1 = open(training_file)
    lines1 = f1.readlines()
    f1.close()
    f2 = open(result_file)
    lines2 = f2.readlines()
    f2.close()
    for i in range(1, len(lines1)):
        id1 = lines1[i].split('\t')[0]
        id2 = lines2[i].split('\t')[0] + '.jpg'
        label1 = lines1[i].split('\t')[-2]
        label2 = lines2[i].split('\t')[-2]
        if id1 != id2:
            print('error', lines1[i], '!=', lines2[i])
        if label1 != label2:
            print(id1, label1, 'to', label2)
            shutil.copy(os.path.join(figure_path, id1), os.path.join('../', label1 + 'to' + label2, id1))


def tsne_for_plot(training_file, result_file, trusted_file, feature_path):
    import os
    f1 = open(training_file)
    lines1 = f1.readlines()
    f1.close()
    f2 = open(result_file)
    lines2 = f2.readlines()
    f2.close()
    f3 = open(trusted_file)
    lines3 = f3.readlines()
    f3.close()
    labels1 = []
    labels2 = []
    labels3 = []
    features = []
    ids = []
    for i in range(1, len(lines1)):
        id1 = lines1[i].split('\t')[0].replace('.jpg', '')
        id2 = lines2[i].split('\t')[0]
        label1 = lines1[i].split('\t')[-2]
        label2 = lines2[i].split('\t')[-2]
        if id1 != id2:
            print('error', lines1[i], '!=', lines2[i])
        labels1.append(1 if label1 == '是' else 0)
        labels2.append(1 if label2 == '是' else 0)
        features.append(DataLoader.get_feature(os.path.join(feature_path, id1 + '.npy')))
        ids.append(id1)
    trusted_num = len(lines3) - 1
    train_num = len(lines1) - 1
    for i in range(1, len(lines3)):
        id3 = lines3[i].split('\t')[0].replace('.jpg', '')
        label3 = lines3[i].split('\t')[-2]
        labels3.append(3 if label3 == '是' else 2)
        features.append(DataLoader.get_feature(os.path.join(feature_path, id3 + '.npy')))
        ids.append(id3)

    features = np.array(features)
    tsne = TSNE(verbose=True)
    position = tsne.fit_transform(features)
    np.save('position.npy', position)
    np.save('feature.npy', features)
    np.save('ids.npy', np.array(ids))
    # np.savez('tsne_result' + version + '.npz', position=position, train_labels=labels1, result_labels=labels2, trusted_labels=labels3)


def prepare_label_for_plot(training_file, result_file, trusted_file, version):
    import os
    f1 = open(training_file)
    lines1 = f1.readlines()
    f1.close()
    f2 = open(result_file)
    lines2 = f2.readlines()
    f2.close()
    f3 = open(trusted_file)
    lines3 = f3.readlines()
    f3.close()
    labels1 = []
    labels2 = []
    labels3 = []
    for i in range(1, len(lines1)):
        id1 = lines1[i].split('\t')[0].replace('.jpg', '')
        id2 = lines2[i].split('\t')[0]
        label1 = lines1[i].split('\t')[-2]
        label2 = lines2[i].split('\t')[-2]
        if id1 != id2:
            print('error', lines1[i], '!=', lines2[i])
        labels1.append(1 if label1 == '是' else 0)
        labels2.append(1 if label2 == '是' else 0)
    trusted_num = len(lines3) - 1
    train_num = len(lines1) - 1
    for i in range(1, len(lines3)):
        id3 = lines3[i].split('\t')[0].replace('.jpg', '')
        label3 = lines3[i].split('\t')[-2]
        labels3.append(3 if label3 == '是' else 2)

    np.savez('tsne_result' + version + '.npz', train_labels=labels1, result_labels=labels2, trusted_labels=labels3)


def cluster_DBSCAN_plot():
    position = np.load('position.npy')
    feature = np.load('feature.npy')
    feature = StandardScaler().fit_transform(feature)
    for eps in [0.1, 0.3, 0.5, 1, 2, 3, 5, 7, 10]:
        for min_samples in [2, 5, 10, 20]:
            print('eps=', eps, 'min_samples=', min_samples)
            db = DBSCAN(eps=eps, min_samples=min_samples).fit(feature)
            core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
            core_samples_mask[db.core_sample_indices_] = True
            labels = db.labels_
            np.save('DBSCAN_labels_eps=' + str(eps) + '_min_samples=' + str(min_samples) + '.npy', labels)

            # Number of clusters in labels, ignoring noise if present.
            n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
            n_noise_ = list(labels).count(-1)

            print('Estimated number of clusters: %d' % n_clusters_)
            print('Estimated number of noise points: %d' % n_noise_)
            print("Silhouette Coefficient: %0.3f"
                  % metrics.silhouette_score(position, labels))

            # #############################################################################
            # Plot result
            import matplotlib.pyplot as plt

            # Black removed and is used for noise instead.
            unique_labels = set(labels)
            colors = [plt.cm.Spectral(each)
                      for each in np.linspace(0, 1, len(unique_labels))]
            for k, col in zip(unique_labels, colors):
                if k == -1:
                    # Black used for noise.
                    col = [0, 0, 0, 1]

                class_member_mask = (labels == k)

                xy = position[class_member_mask & core_samples_mask]
                plt.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=tuple(col),
                         markersize=6)

                xy = position[class_member_mask & ~core_samples_mask]
                plt.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=tuple(col),
                         markersize=4)

            plt.title('Estimated number of clusters: %d' % n_clusters_)
            plt.savefig('DBSCAN_plot_eps=' + str(eps) + '_min_samples=' + str(min_samples) + '.png')
            plt.close()


def cluster_KMeans_plot():
    position = np.load('position.npy')
    feature = np.load('feature.npy')
    feature = StandardScaler().fit_transform(feature)
    for n_clusters in [30, 50, 70, 100, 120, 150, 170, 200]:
        print('n_clusters=', n_clusters)
        km = KMeans(n_clusters=n_clusters, random_state=0).fit(feature)
        labels = km.labels_
        np.save('KMeans_labels_n_clusters=' + str(n_clusters) + '.npy', labels)

        # Number of clusters in labels, ignoring noise if present.
        n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
        n_noise_ = list(labels).count(-1)

        print('Estimated number of clusters: %d' % n_clusters_)
        print('Estimated number of noise points: %d' % n_noise_)
        print("Silhouette Coefficient: %0.3f"
              % metrics.silhouette_score(position, labels))

        # #############################################################################
        # Plot result
        import matplotlib.pyplot as plt

        # Black removed and is used for noise instead.
        unique_labels = set(labels)
        colors = [plt.cm.Spectral(each)
                  for each in np.linspace(0, 1, len(unique_labels))]
        for k, col in zip(unique_labels, colors):
            if k == -1:
                # Black used for noise.
                col = [0, 0, 0, 1]

            class_member_mask = (labels == k)

            xy = position[class_member_mask]
            plt.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=tuple(col),
                     markersize=6)

            xy = position[class_member_mask]
            plt.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=tuple(col),
                     markersize=4)

        plt.title('Estimated number of clusters: %d' % n_clusters_)
        plt.savefig('KMeans_plot_n_clusters=' + str(n_clusters) + '.png')
        plt.close()


def draw_plot(version):
    data = np.load('tsne_result' + version + '.npz')
    position = np.load('position.npy')
    labels1 = data['train_labels']
    labels2 = data['result_labels']
    labels3 = data['trusted_labels']
    color = ['green', 'blue', 'red', 'yellow']
    labels = np.concatenate((labels1, labels3))
    selection0 = labels == 0
    selection1 = labels == 1
    selection2 = labels == 2
    selection3 = labels == 3
    l0 = plt.scatter(position[selection0][:, 0], position[selection0][:, 1], 6, c=color[0])
    l1 = plt.scatter(position[selection1][:, 0], position[selection1][:, 1], 6, c=color[1])
    l2 = plt.scatter(position[selection2][:, 0], position[selection2][:, 1], 4, c=color[2])
    l3 = plt.scatter(position[selection3][:, 0], position[selection3][:, 1], 4, c=color[3])
    plt.legend(handles=[l0, l1, l2, l3], labels=['no', 'yes', 'trusted-no', 'trusted-yes'], loc='best')
    # plt.legend(handles=[l0, l1], labels=['no', 'yes'], loc='best')
    plt.savefig('train_plot' + version + '.png')
    plt.close()

    labels = np.concatenate((labels2, labels3))
    selection0 = labels == 0
    selection1 = labels == 1
    selection2 = labels == 2
    selection3 = labels == 3
    l0 = plt.scatter(position[selection0][:, 0], position[selection0][:, 1], 6, c=color[0])
    l1 = plt.scatter(position[selection1][:, 0], position[selection1][:, 1], 6, c=color[1])
    l2 = plt.scatter(position[selection2][:, 0], position[selection2][:, 1], 4, c=color[2])
    l3 = plt.scatter(position[selection3][:, 0], position[selection3][:, 1], 4, c=color[3])
    plt.legend(handles=[l0, l1, l2, l3], labels=['no', 'yes', 'trusted-no', 'trusted-yes'], loc='best')
    # plt.legend(handles=[l0, l1], labels=['no', 'yes'], loc='best')
    plt.savefig('result_plot' + version + '.png')
    plt.close()


def draw_plot1(version):
    data = np.load('tsne_result' + version + '.npz')
    position = np.load('position.npy')
    labels1 = data['train_labels']
    labels2 = data['result_labels']
    labels3 = data['trusted_labels']
    color = ['green', 'blue', 'red', 'yellow']
    labels = np.concatenate((labels1, labels3))
    selection0 = labels == 0
    selection1 = labels == 1
    # selection2 = labels == 2
    # selection3 = labels == 3
    l0 = plt.scatter(position[selection0][:, 0], position[selection0][:, 1], 6, c=color[0])
    l1 = plt.scatter(position[selection1][:, 0], position[selection1][:, 1], 6, c=color[1])
    # l2 = plt.scatter(position[selection2][:, 0], position[selection2][:, 1], 4, c=color[2])
    # l3 = plt.scatter(position[selection3][:, 0], position[selection3][:, 1], 4, c=color[3])
    # plt.legend(handles=[l0, l1, l2, l3], labels=['no', 'yes', 'trusted-no', 'trusted-yes'], loc='best')
    plt.legend(handles=[l0, l1], labels=['no', 'yes'], loc='best')
    plt.savefig('train_plot1' + version + '.png')
    plt.close()

    labels = np.concatenate((labels2, labels3))
    selection0 = labels == 0
    selection1 = labels == 1
    selection2 = labels == 2
    selection3 = labels == 3
    l0 = plt.scatter(position[selection0][:, 0], position[selection0][:, 1], 6, c=color[0])
    l1 = plt.scatter(position[selection1][:, 0], position[selection1][:, 1], 6, c=color[1])
    # l2 = plt.scatter(position[selection2][:, 0], position[selection2][:, 1], 4, c=color[2])
    # l3 = plt.scatter(position[selection3][:, 0], position[selection3][:, 1], 4, c=color[3])
    # plt.legend(handles=[l0, l1, l2, l3], labels=['no', 'yes', 'trusted-no', 'trusted-yes'], loc='best')
    plt.legend(handles=[l0, l1], labels=['no', 'yes'], loc='best')
    plt.savefig('result_plot1_' + version + '.png')
    plt.close()


def draw_plot2(position, label, filename):
    color = ['blue', 'red']
    selection0 = label == 0
    selection1 = label == 1
    l0 = plt.scatter(position[selection0][:, 0], position[selection0][:, 1], 6, c=color[0])
    l1 = plt.scatter(position[selection1][:, 0], position[selection1][:, 1], 6, c=color[1])
    plt.legend(handles=[l0, l1], labels=['yes', 'no'], loc='best')
    plt.savefig(filename)
    plt.close()


def draw_plot3(position, label, filename):
    color = ['blue', 'green', 'red', 'yellow']
    selection0 = label == 0
    selection1 = label == 1
    selection2 = label == 2
    selection3 = label == 3
    l0 = plt.scatter(position[selection0][:, 0], position[selection0][:, 1], 6, c=color[0])
    l1 = plt.scatter(position[selection1][:, 0], position[selection1][:, 1], 6, c=color[1])
    l2 = plt.scatter(position[selection2][:, 0], position[selection2][:, 1], 6, c=color[2])
    l3 = plt.scatter(position[selection3][:, 0], position[selection3][:, 1], 6, c=color[3])
    plt.legend(handles=[l0, l1, l2, l3], labels=['game', 'other', 'trust game', 'trust other'], loc='best')
    plt.savefig(filename)
    plt.close()


def create_dataset_for_duti_from_finetune_result(feature_dir, target_dataset, version=''):
    features = np.load(feature_dir + '/noisy_item_' + version + 'last_but_one_output.npy').tolist()
    trusted_feature1 = np.load(feature_dir + '/train_' + version + 'last_but_one_output.npy')
    N1, D = trusted_feature1.shape
    trusted_feature2 = np.load(feature_dir + '/validation_add_trust_' + version + 'last_but_one_output.npy')
    N2, D = trusted_feature2.shape
    paths = np.load(feature_dir + '/noisy_item_' + version + 'last_but_one_output_ids.npy')
    trusted_ids1 = np.load(feature_dir + '/train_' + version + 'last_but_one_output_ids.npy')
    trusted_ids2 = np.load(feature_dir + '/validation_add_trust_' + version + 'last_but_one_output_ids.npy')
    trusted_confidence1 = np.load(feature_dir + '/train_' + version + 'outputs.npy')
    trusted_confidence2 = np.load(feature_dir + '/validation_add_trust_' + version + 'outputs.npy')
    ground_truth = np.load('../data/ground_truth_201909111829.npz')
    gt_ids = ground_truth['ids']
    gt_labels = ground_truth['labels']

    gt_label = []
    label = []
    ids = []
    feature = []
    for i in range(len(paths)):
        if paths[i].split('/')[1].split('.')[0] not in ids:
            ids.append(paths[i].split('/')[1].split('.')[0])
            feature.append(features[i])
            name = paths[i].split('/')[1]
            index = np.where(gt_ids == name)[0][0]
            gt_label.append(gt_labels[index])
            if 'game' in paths[i]:
                label.append(0)
            else:
                label.append(1)
    feature = np.array(feature)
    trusted_ids = []
    trusted_label = []
    trusted_feature = []
    trusted_confidence = []
    for i in range(len(trusted_ids1)):
        if trusted_ids1[i].split('/')[1].split('.')[0] not in trusted_ids:
            l = 0 if 'game' in trusted_ids1[i] else 1
            if abs(l - trusted_confidence1[i]) < 1:
                trusted_label.append(l)
                trusted_ids.append(trusted_ids1[i].split('/')[1].split('.')[0])
                trusted_feature.append(trusted_feature1[i])
                trusted_confidence.append(1 - abs(l - trusted_confidence1[i]))
    for i in range(len(trusted_ids2)):
        if trusted_ids2[i].split('/')[1].split('.')[0] not in trusted_ids:
            l = 0 if 'game' in trusted_ids2[i] else 1
            if abs(l - trusted_confidence2[i]) < 1:
                trusted_label.append(l)
                trusted_ids.append(trusted_ids2[i].split('/')[1].split('.')[0])
                trusted_feature.append(trusted_feature2[i])
                trusted_confidence.append(1 - abs(l - trusted_confidence2[i]))
    print('chose', len(trusted_ids), 'from', N1 + N2)
    dup_ids = []
    noisy_num = len(ids)
    for i in range(len(trusted_ids)):
        if trusted_ids[i] not in ids:
            ids.append(trusted_ids[i])
            label.append(trusted_label[i] + 0)
            feature.append(trusted_feature[i])

            index = np.where(gt_ids == trusted_ids[i])[0][0]
            gt_label.append(gt_labels[index])
        else:
            dup_ids.append(trusted_ids[i])
    total_num = len(ids)
    selection = np.zeros(total_num, dtype=bool)
    selection[noisy_num:] = 1
    for i in range(len(ids)):
        if ids[i] in dup_ids:
            label[i] += 0
            selection[i] = 1
    feature = np.array(feature)
    trusted_feature = np.array(trusted_feature)
    label = np.array(label)
    trusted_label = np.array(trusted_label)
    ids = np.array(ids)
    trusted_ids = np.array(trusted_ids)
    trusted_confidence = np.array(trusted_confidence)

    from sklearn.decomposition import PCA
    pca = PCA(n_components=128)
    feature_pca = pca.fit_transform(feature)
    trusted_feature_pca = pca.fit_transform(trusted_feature)

    np.savez('../data/' + target_dataset + '.npz', trusted_confidence=trusted_confidence, gt_label=gt_label, selection=selection, ids=ids, trusted_ids=trusted_ids, feature=feature, label=label, trusted_feature=trusted_feature, trusted_label=trusted_label)
    np.savez('../data/' + target_dataset + '_pca_128.npz', trusted_confidence=trusted_confidence, gt_label=gt_label, selection=selection, ids=ids, trusted_ids=trusted_ids, feature=feature_pca, label=label, trusted_feature=trusted_feature_pca, trusted_label=trusted_label)


def clear_dir(path):
    import os
    if not os.path.exists(path):
        os.makedirs(path)
        return
    files = os.listdir(path)
    for file in files:
        os.remove(path + '/' + file)


def create_game_trusted2_3class_plan_C():
    import numpy as np
    data = np.load('../data/game_trusted2_3class.npz')
    valid_count = len(data['valid_ids'])
    test_ids = data['test_ids']
    test_trusted_labels = data['test_trusted_labels']
    test_noisy_labels = data['test_noisy_labels']
    test_features = data['test_features']
    selection = np.random.choice(len(test_ids), len(test_ids), replace=False)
    np.savez('../data/game_trusted2_3class_plan_C.npz',
             features=data['features'],
             ids=data['ids'],
             labels=data['labels'],
             train_features=data['train_features'],
             train_ids=data['train_ids'],
             train_trusted_labels=data['train_trusted_labels'],
             train_noisy_labels=data['train_noisy_labels'],
             valid1_features=data['valid_features'],
             valid1_ids=data['valid_ids'],
             valid1_trusted_labels=data['valid_trusted_labels'],
             valid1_noisy_labels=data['valid_noisy_labels'],
             valid2_features=data['test_features'][selection[:valid_count]],
             valid2_ids=data['test_ids'][selection[:valid_count]],
             valid2_trusted_labels=data['test_trusted_labels'][selection[:valid_count]],
             valid2_noisy_labels=data['test_noisy_labels'][selection[:valid_count]],
             test_features=data['test_features'][selection[valid_count:]],
             test_ids=data['test_ids'][selection[valid_count:]],
             test_trusted_labels=data['test_trusted_labels'][selection[valid_count:]],
             test_noisy_labels=data['test_noisy_labels'][selection[valid_count:]])


def run_tsne_and_save_plot(feature, label, label_names, colors, path):
    from sklearn.manifold import TSNE
    import numpy as np
    tsne = TSNE(n_components=2)
    position = tsne.fit_transform(feature)
    np.save(path + '_position.npy', position)
    ls = []
    for i in range(len(label_names)):
        ls.append(plt.scatter(position[label == i][:, 0], position[label == i][:, 1], 5, c=colors[i]))
    plt.legend(handles=ls, labels=list(label_names), loc='best')
    plt.savefig(path + '_position.png')
    plt.close()


def get_class_distribution(label):
    distribution = []
    names = []
    for l in label:
        if l not in names:
            names.append(l)
            distribution.append(0)
        distribution[names.index(l)] += 1
    print(names)
    print(distribution)


def full_train_in_game_1m_batchs_npz_for_duti():
    import numpy as np
    for num_class in [6, 15]:
        train_features = []
        train_ids = []
        train_noisy_labels = []
        train_trusted_labels = []
        for batch_id in range(21):
            data = np.load('../data/all_batchs/game_1m_' + str(num_class) + 'class_batch' + str(batch_id) + '.npz')
            train_ids.append(data['train_ids'])
            train_features.append(data['train_features'])
            train_noisy_labels.append(data['train_noisy_labels'])
            train_trusted_labels.append(data['train_trusted_labels'])
        train_features = np.concatenate(train_features)
        train_ids = np.concatenate(train_ids)
        train_noisy_labels = np.concatenate(train_noisy_labels)
        train_trusted_labels = np.concatenate(train_trusted_labels)
        for batch_id in range(21):
            data = np.load('../data/all_batchs/game_1m_' + str(num_class) + 'class_batch' + str(batch_id) + '.npz')
            np.savez('../data/all_batchs/game_1m_' + str(num_class) + 'class_full_train_batch' + str(batch_id) + '.npz',
                    ids=data['ids'],
                    features=data['features'],
                    labels=data['labels'],
                    train_ids=train_ids,
                    train_features=train_features,
                    train_noisy_labels=train_noisy_labels,
                    train_trusted_labels=train_trusted_labels,
                    test_ids=data['test_ids'],
                    test_features=data['test_features'],
                    test_noisy_labels=data['test_noisy_labels'],
                    test_trusted_labels=data['test_trusted_labels'],
                    valid_ids=data['valid_ids'],
                    valid_features=data['valid_features'],
                    valid_noisy_labels=data['valid_noisy_labels'],
                    valid_trusted_labels=data['valid_trusted_labels'])


def change_predicyion_in_game_1m_batchs_npz_for_duti():
    import numpy as np
    for batch_id in range(21):
        data = np.load('../data/all_batchs/game_1m_15class_full_train_batch' + str(batch_id) + '.npz')
        predictions = np.load('../data/all_batchs/all_full_train_prediction_labels_batchs/game_1m_15class_prediction_labels_batch' + str(batch_id) + '.npy')
        np.savez('../data/all_batchs/game_1m_15class_full_train_predicted_batch' + str(batch_id) + '.npz',
                ids=data['ids'],
                features=data['features'],
                labels=predictions.reshape(-1),
                train_ids=data['train_ids'],
                train_features=data['train_features'],
                train_noisy_labels=data['train_noisy_labels'],
                train_trusted_labels=data['train_trusted_labels'],
                test_ids=data['test_ids'],
                test_features=data['test_features'],
                test_noisy_labels=data['test_noisy_labels'],
                test_trusted_labels=data['test_trusted_labels'],
                valid_ids=data['valid_ids'],
                valid_features=data['valid_features'],
                valid_noisy_labels=data['valid_noisy_labels'],
                valid_trusted_labels=data['valid_trusted_labels'])


if __name__ == '__main__':
    # create dataset for duti
    # full_train_in_game_1m_batchs_npz_for_duti()
    change_predicyion_in_game_1m_batchs_npz_for_duti()
    for i in range(21):
        transform_npz_to_mat('../data/all_batchs/game_1m_15class_predicted_batch' + str(i) + '.npz',
                         '../data/all_batchs/game_1m_15class_predicted_batch' + str(i) + '.mat')
    # transform_mat_to_npz('../DUTI_code/greedy_result_1.mat',
    #                      '../DUTI_code/greedy_result_1.npz')
    exit(1)
    import numpy as np
    for name in ['valid']:
        for t in ['trusted', 'noisy']:
            print(name + '_' + t)
            data = np.load('../data/game_trusted2_3class.npz')
            # get_class_distribution(data[name + '_' + t + '_labels'])
            run_tsne_and_save_plot(data[name + '_' + 'features'],
                                   data[name + '_' + t + '_labels'],
                                   ['wzry', 'cjzc', 'other'],
                                   ['green', 'blue', 'red'],
                                   '../data/' + name + '_' + t)
    # create_dataset_for_duti_from_finetune_result('../data/feature201909091410', 'Compressed_for_vis_2522_finetune_201909091410')
    # create_dataset_for_duti_from_finetune_result('../data/feature201908282134', 'Compressed_for_vis_2522_finetune_201908312250')
    # create_game_trusted2_3class_plan_C()
    exit(1)


    # process for duti
    # transform_npz_to_mat('../data/Compressed_for_vis_2522_finetune_201909091410_pca_128.npz', '../data/Compressed_for_vis_2522_finetune_201909091410_pca_128.mat')
    # transform_npz_to_mat('../data/Compressed_for_vis_2522_finetune_201908312250.npz', '../data/Compressed_for_vis_2522_finetune_201908312250.mat')
    # transform_npz_to_mat('../data/Compressed_for_vis_2522_finetune2_201908312250_pca_128.npz', '../data/Compressed_for_vis_2522_finetune2_201908312250_pca_128.mat')
    # transform_npz_to_mat('../data/Compressed_for_vis_2522_finetune2_201908312250.npz', '../data/Compressed_for_vis_2522_finetune2_201908312250.mat')
    # exit(1)

    # process result of duti
    # transform_mat_to_npz('../DUTI_code/greedy_result-1-finetune-for_vis_201909091410_pca_128.mat',
    #                      '../result/greedy_result-1-finetune-for_vis_201909091410_pca_128.npz')
    # source_data = np.load('../data/Compressed_for_vis_2522_finetune_201909091410_pca_128.npz')
    # result = np.load('../result/greedy_result-1-finetune-for_vis_201909091410_pca_128.npz')
    #
    # for r in range(8):
    #     sel = result['rankings'].reshape(-1) == r
    #     la = source_data['label'][sel]
    #     gt_la = source_data['gt_label'][sel]
    #     t_la = result['y_debug'].reshape(-1)
    #     # t_la[source_data['selection']] = source_data['trusted_label']
    #     t_la = t_la[sel]
    #     count = [0, 0, 0, 0]
    #     for i in range(len(t_la)):
    #         if t_la[i] == 0 and la[i] == 1:
    #             count[0] += 1
    #             if t_la[i] == gt_la[i]:
    #                 count[2] += 1
    #         elif t_la[i] == 1 and la[i] == 0:
    #             count[1] += 1
    #             if t_la[i] == gt_la[i]:
    #                 count[3] += 1
    #     print('ranking=', r, count)
    # # exit(1)
    #
    # acc = source_data['label'] == source_data['gt_label']
    # print('init:', sum(acc), '/', len(acc), '=', sum(acc) / len(acc))
    # acc[source_data['selection']] = True
    # print('add trust:', sum(acc), '/', len(acc), '=', sum(acc) / len(acc))
    #
    # finetune_label = result['y_debug'].reshape(-1)
    # acc = source_data['gt_label'] == finetune_label
    # acc[source_data['selection']] = True
    # print('duti:', sum(acc), '/', len(acc), '=', sum(acc) / len(acc))
    # finetune_label[source_data['selection']] = source_data['gt_label'][source_data['selection']]
    # np.save('finetune_label-for_vis_201909091410_pca_128.npy', finetune_label)
    # transform_mat_to_npz('../DUTI_code/greedy_result-1-finetune2-for_vis_201908312250_pca_128.mat',
    #                      '../result/greedy_result-1-finetune2-for_vis_201908312250_pca_128.npz')
    # source_data = np.load('../data/Compressed_for_vis_2522_finetune2_201908312250_pca_128.npz')
    # result = np.load('../result/greedy_result-1-finetune2-for_vis_201908312250_pca_128.npz')
    # finetune2_label = result['y_debug'].reshape(-1)
    # rankings = result['rankings'].reshape(-1) > 3
    # finetune2_label[rankings] = source_data['label'][rankings]
    # finetune2_label[source_data['selection']] = source_data['trusted_label']
    # np.save('finetune2_label-for_vis_201908312250_pca_128.npy', finetune2_label)
    # exit(1)

    # tsne to get position
    # tsne = TSNE(verbose=True)
    # source_data = np.load('../data/Compressed_for_vis_2522_finetune_201909091410_pca_128.npz')
    # position = tsne.fit_transform(source_data['feature'])
    # np.save('position_for_vis_2522_finetune_201909091410_pca_128.npy', position)
    # source_data = np.load('../data/Compressed_for_vis_2522_finetune2_201908312250_pca_128.npz')
    # position = tsne.fit_transform(source_data['feature'])
    # np.save('position_for_vis_2522_finetune2_201908312250_pca_128.npy', position)
    # exit(1)

    # draw plot of tsne
    source_data = np.load('../data/Compressed_for_vis_2522_finetune_201909091410_pca_128.npz')
    matlab_label = np.load('finetune_label-for_vis_201909091410_pca_128.npy')
    position = np.load('position_for_vis_2522_finetune_201909091410_pca_128.npy')
    draw_plot2(position, source_data['label'], 'before_finetune_128.png')
    draw_plot2(position, matlab_label, 'after_finetune_128.png')
    # source_data = np.load('../data/Compressed_for_vis_2522_finetune2_201908312250_pca_128.npz')
    # matlab_label = np.load('finetune2_label-for_vis_201908312250_pca_128.npy')
    # position = np.load('position_for_vis_2522_finetune2_201908312250_pca_128.npy')
    # draw_plot2(position, source_data['label'], 'before_finetune2_128.png')
    # draw_plot2(position, matlab_label, 'after_finetune2_128.png')
    # exit(1)

    # copy file to proper directory
    # source_data = np.load('../data/Compressed_for_vis_2522_finetune3_201909012213_pca_128.npz')
    # matlab_label = np.load('finetune3_label-for_vis_201909012213_pca_128.npy')
    # ids = np.load('../data/feature201909012213/noisy_item_processed_last_but_one_output_ids.npy')
    # rankings = np.load('../result/greedy_result-1-finetune3-for_vis_201909012213_pca_128.npz')['rankings'].reshape(-1)
    # import shutil
    # count = [0, 0, 0, 0]
    # for i in range(len(ids)):
    #     if source_data['label'][i] == 0 and matlab_label[i] == 1:
    #         shutil.copy('../forTHU/' + ids[i].split('/')[1], '../finetune3/是to否/' + str(int(rankings[i])) + '/' + ids[i].split('/')[1])
    #         count[1] += 1
    #     elif source_data['label'][i] == 1 and matlab_label[i] == 0:
    #         shutil.copy('../forTHU/' + ids[i].split('/')[1], '../finetune3/否to是/' + str(int(rankings[i])) + '/' + ids[i].split('/')[1])
    #         count[2] += 1
    #     elif source_data['label'][i] == 0:
    #         count[0] += 1
    #     else:
    #         count[3] += 1
    # print(count)
    # source_data = np.load('../data/Compressed_for_vis_2522_finetune2_201908312250_pca_128.npz')
    # matlab_label = np.load('finetune2_label-for_vis_201908312250_pca_128.npy')
    # ids = np.load('../data/feature201908300204/noisy_item_last_but_one_output_ids.npy')
    #
    # rankings = np.load('../result/greedy_result-1-finetune2-for_vis_201908312250_pca_128.npz')['rankings'].reshape(-1)
    # import shutil
    # count = [0, 0, 0, 0]
    # for i in range(len(ids)):
    #     if source_data['label'][i] == 0 and matlab_label[i] == 1:
    #         shutil.copy('../forTHU/' + ids[i].split('/')[1], '../finetune2/是to否/' + ids[i].split('/')[1])
    #         count[1] += 1
    #     elif source_data['label'][i] == 1 and matlab_label[i] == 0:
    #         shutil.copy('../forTHU/' + ids[i].split('/')[1], '../finetune2/否to是/' + ids[i].split('/')[1])
    #         count[2] += 1
    #     elif source_data['label'][i] == 0:
    #         # shutil.copy('../forTHU/' + ids[i].split('/')[1], '../finetune2/是to是/' + ids[i].split('/')[1])
    #         count[0] += 1
    #     else:
    #         # shutil.copy('../forTHU/' + ids[i].split('/')[1], '../finetune2/否to否/' + ids[i].split('/')[1])
    #         count[3] += 1
    #     # print(i, count)
    # print(count)
    # exit(1)

    # create dataset for DQAnalyzer
    source_data = np.load('../data/Compressed_for_vis_2522_finetune_201909091410_pca_128.npz')
    matlab_label = np.load('finetune_label-for_vis_201909091410_pca_128.npy')
    position = np.load('position_for_vis_2522_finetune_201909091410_pca_128.npy')
    X = position
    ids = source_data['ids']
    label = source_data['label']
    gt_label = source_data['gt_label']
    label[source_data['selection']] += 2# source_data['trusted_label'] + 2
    gt_label[source_data['selection']] += 2
    label1 = matlab_label
    label1[source_data['selection']] += 2 #source_data['trusted_label'] + 2
    label_names = ['游戏', '其他', '可信点游戏', '可信点其他']
    changed_item = []
    wrong_item = []
    # count = [0, 0, 0, 0]
    for i in range(len(label)):
        # count[label[i] * 2 + gt_label[i]] += 1
        if label[i] != label1[i]:
            changed_item.append({
                'id': i,
                'name': ids[i],
                'label': int(label[i])
            })
        if label[i] != gt_label[i]:
            wrong_item.append({
                'id': i,
                'name': ids[i],
                'label': int(label[i])
            })
    # print(count)
    import json

    with open('finetune_changed_item.json', 'w', encoding='utf-8') as f:
        f.write(json.dumps(changed_item))
        f.close()
    with open('finetune_wrong_item.json', 'w', encoding='utf-8') as f:
        f.write(json.dumps(wrong_item))
        f.close()
    np.savez('finetune_game_noisy_trusted.npz', X=X, ids=ids, label=label, label1=label1,
             label_names=np.array(label_names))

    position = np.load('position_for_vis_2522_finetune_201909091410_pca_128.npy')
    draw_plot3(position, label, 'before_vis_128.png')
    draw_plot3(position, label1, 'after_vis_128.png')

    # source_data = np.load('../data/Compressed_for_vis_2522_finetune2_201908312250_pca_128.npz')
    # matlab_label = np.load('finetune2_label-for_vis_201908312250_pca_128.npy')
    # position = np.load('position_for_vis_2522_finetune2_201908312250_pca_128.npy')
    # X = position
    # ids = source_data['ids']
    # label = source_data['label']
    # label[source_data['selection']] = source_data['trusted_label'] + 2
    # label1 = matlab_label
    # label1[source_data['selection']] = source_data['trusted_label'] + 2
    # label_names = ['游戏', '其他', '可信点游戏', '可信点其他']
    # changed_item = []
    # for i in range(len(label)):
    #     if label[i] != label1[i]:
    #         changed_item.append({
    #             'id': i,
    #             'label': int(label[i])
    #         })
    # import json
    # with open('finetune2_changed_item.json', 'w', encoding='utf-8') as f:
    #     f.write(json.dumps(changed_item))
    #     f.close()
    # np.savez('finetune2_game_noisy_trusted.npz', X=X, ids=ids, label=label, label1=label1, label_names=np.array(label_names))
    print('finished')
    exit(1)


# python_label = python_result['y_debug'].reshape(-1)
    # count = 0
    # for i in range(len(matlab_label)):
    #     if (matlab_label[i] == python_label[i]):
    #         count += 1
    #     else:
    #         print(i)
    # print('accuracy:', count / len(matlab_label))
    # tsne_for_plot('../data/Train_2522.txt', '../result/Result1_2522.txt', '../data/Trusted_2522.txt',
    #               '../data/game_features')
    # exit(1)
    # versions = ['conf=1', 'conf=100', 'method=dt_default', 'method=dt_min_leaf=5', 'method=dt_min_split_max_depth',
    #             'method=dt_min_split', 'method=dt', 'method=dt_min_leaf=2']
    # for version in versions:
    #     prepare_label_for_plot('../data/Train_2522.txt', '../result/Result_2522_' + version + '.txt', '../data/Trusted_2522.txt', version)
    #     draw_plot(version)
    #     draw_plot1(version)
    # check_which_be_correct('../data/Train_2522.txt', '../result/Result_2522_method=dt_min_leaf=2.txt', '../forTHU/')
    # cluster_DBSCAN_plot()
    # cluster_KMeans_plot()