"""
Save result
Created by shouxing, 2019/6/27
"""

import numpy as np


def save_as_txt(path, data):
    """
    This function aims to save data as the specified format.
    :param path: path to save the file.
    :param data: source data which would be written into the file.
    :return: None
    """
    f = open(path, 'w')
    if len(data) == 0:
        f.close()
        return
    else:
        f.write('\t'.join(['Result', 'Multi' if data[0]['flag'] == 0 else 'Single']) + '\n')

    for item in data:
        if item['flag'] == 0:
            f.write('\t'.join([item['id'], item['frame'], '默认分类', item['label_name'], item['other']]) + '\n')
        else:
            f.write('\t'.join([item['id'], item['frame'], item['label_name'], '是' if item['flag'] == 1 else '否', item['other']]) + '\n')
    f.close()


def save_as_json(path, data):
    """
    # TODO add some description for this function
    :param path:
    :param data:
    :return:
    """
    # TODO finish the code for this function
    pass


if __name__ == '__main__':
    from src import DataLoader
    flag, dataset = DataLoader.load_from_txt("../data/data1.txt")
    save_as_txt("../result/result1.txt")
