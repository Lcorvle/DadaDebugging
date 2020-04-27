"""
Load data from the origin file
Created by shouxing, 2019/6/27
"""

import numpy as np
import os


def load_from_txt(path):
    """
    Load information from a txt file.
    :param path: the path of the txt file.
    :return: this function will return a string and a list, each item of the list is a dictionary
                include 5 keys:
                    id,
                    frame,
                    label_name,
                    flag(0: multi-class, 1: binary-class 是, 2: binary-class 否),
                    other.
    """
    f = open(path)
    lines = f.readlines()
    f.close()
    if len(lines) == 0:
        return None, np.array([[]])
    elif len(lines) == 1:
        return lines[0].replace('\n', ''), np.array([[]])
    result = []
    flags = lines[0].replace('\n', '').split('\t')
    flag = flags[1] == 'Multi'
    for line in lines[1:]:
        line = line.replace('\n', '')
        vals = line.split('\t')
        if flag:
            result.append({
                'id': vals[0].replace('.jpg', ''),
                'frame': vals[1],
                'label_name': vals[3],
                'flag': 0,
                'other': vals[4]
            })
        else:
            result.append({
                'id': vals[0].replace('.jpg', ''),
                'frame': vals[1],
                'label_name': vals[2],
                'flag': 1 if vals[3] == '是' else 2,
                'other': vals[4]
            })
    return flags[0], result


def get_feature(path):
    """
    Get the feature of a specified item.
    :param id: the id of the item.
    :return: a feature vector as type of np.ndarray.
    """
    if os.path.exists(path):
        return np.load(path).reshape(-1)
    else:
        # return np.random.random(32)
        raise FileNotFoundError


def merge_dataset_and_trusted_items(data, trusted_items):
    """
    Merge dataset and the trust items.
    :param data: a list, each item of the list is a dictionary
                include 3 keys:
                    id,
                    label_name,
                    flag(0: multi-class, 1: binary-class 是, 2: binary-class 否).
    :param trusted_items: a list, each item of the list is a dictionary
                include 3 keys:
                    id,
                    label_name,
                    flag(0: multi-class, 1: binary-class 是, 2: binary-class 否).
    :return: res: a dictionary or a list
             label_names: a list
             flag: boolean
             other: string

             option1: res: a dictionary include 5 keys:
                        labels, ids, others, trusted_ids, trusted_labels
                      label_names: a list of label_names
                      flag: True
             option2: res: a list, each item of it is a dictionary like the res in the option1
                      label_names: a list of label_names
                      flag: False

    """
    if len(data) == 0:
        return [], [], True
    if data[0]['flag'] == 0:
        label_names = []
        res = {
            'labels': [],
            'ids': [],
            'others': [],
            'frames': [],
            'trusted_ids': [],
            'trusted_labels': []
        }
        for item in data:
            if item['label_name'] not in label_names:
                label_names.append(item['label_name'])
            res['ids'].append(item['id'])
            res['others'].append(item['other'])
            res['frames'].append(item['frame'])
            res['labels'].append(label_names.index(item['label_name']))

        for item in trusted_items:
            if item['label_name'] not in label_names:
                continue
            if item['flag'] < 2:
                res['trusted_ids'].append(item['id'])
                res['trusted_labels'].append(label_names.index(item['label_name']))
    else:
        res = []
        label_names = []
        for item in data:
            if item['label_name'] not in label_names:
                label_names.append(item['label_name'])
                res.append({
                    'labels': [],
                    'ids': [],
                    'others': [],
                    'frames': [],
                    'trusted_ids': [],
                    'trusted_labels': []
                })
            res[label_names.index(item['label_name'])]['ids'].append(item['id'])
            res[label_names.index(item['label_name'])]['others'].append(item['other'])
            res[label_names.index(item['label_name'])]['frames'].append(item['frame'])
            res[label_names.index(item['label_name'])]['labels'].append(item['flag'] - 1)
        for item in trusted_items:
            if item['label_name'] not in label_names:
                continue

            res[label_names.index(item['label_name'])]['trusted_ids'].append(item['id'])
            if item['flag'] == 0:
                res[label_names.index(item['label_name'])]['trusted_labels'].append(0)
            else:
                res[label_names.index(item['label_name'])]['trusted_labels'].append(item['flag'] - 1)
    return res, label_names, data[0]['flag'] == 0


def load_from_json(path):
    """
    #TODO finish the function for loading json files
    load file from json file.
    :param path:
    :return:
    """
    return np.zeros(1)


if __name__ == '__main__':
    print("test the function load_from_txt")
    flag, dataset = load_from_txt('../data/data1.txt')
    if flag != 'Train':
        print('warning: the ../data/data1.txt is not a train file.')
    flag, trusted_items = load_from_txt('../data/trust1.txt')
    if flag != 'Trusted':
        print('warning: the ../data/trust1.txt is not a trusted file.')
    data, labelnames, flag = merge_dataset_and_trusted_items(dataset, trusted_items)
    print(data, labelnames, flag)
    feature = get_feature('9742745727_h')
    print(feature)