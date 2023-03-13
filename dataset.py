import os

import numpy
import numpy as np
import yaml


def fix_label_train():
    # 打开COCO配置文件
    with open('../data/coco_kpts.yaml', encoding='UTF-8') as f:
        data_dict = yaml.safe_load(f)  # data dict
    # 获取父路径
    label_train_parent_path = os.path.join('../', data_dict['label_train'])
    # 获取文件名列表
    label_train_name_list = os.listdir(label_train_parent_path)
    # 获取文件路径名列表（父路径+文件名）
    label_train_file_path_list = []
    for name in label_train_name_list:
        label_train_file_path_list.append(os.path.join(label_train_parent_path + '/', name))
    # 依次打开label文件
    for file in label_train_file_path_list:
        # 打开文件，读
        with open(file, 'r', encoding='UTF-8') as f:
            train_labels = [x.split() for x in f.read().strip().splitlines()]  # 每组标签保存到一个维度（x，56）
            train_labels = np.array(train_labels, dtype=np.float32)  # 转为numpy数组
            fixed_train_labels = []  # 存放修改后的标签数据
            # 对每组标签，删除最后12个数据，即后4个关键点的数据
            for i in range(len(train_labels)):
                fixed_train_label = np.delete(train_labels[i], np.arange(-13, -1))
                fixed_train_labels.append(fixed_train_label)  # 删除后的新数组追加到一起，构成一个新列表
            fixed_train_labels = np.array(fixed_train_labels, dtype=np.float32)  # 转为numpy数组
            print("文件路径：" + file + ',' + "修改前:" + train_labels.shape + ',' + "修改后:" + fixed_train_labels.shape)
            f.close()
        # 打开文件，写
        with open(file, "w") as f:
            # 分别写入每一组标签
            for i in range(len(fixed_train_labels)):
                cls = True  # 标志当前数是否是类别位，类别位是整数，其他位保留6位小数
                for num in fixed_train_labels[i]:
                    if cls:
                        num = int(num)
                    elif not cls:
                        num = format(num, '.6f')
                    cls = False
                    f.write(str(num) + ' ')  # 每个数字用空格隔开
                f.write('\n')  # 每组标签用换行符隔开
            f.close()


def fix_label_val():
    # 打开COCO配置文件
    with open('../data/coco_kpts.yaml', encoding='UTF-8') as f:
        data_dict = yaml.safe_load(f)  # data dict
    # 获取父路径
    label_val_parent_path = os.path.join('../', data_dict['label_val'])
    # 获取文件名列表
    label_val_name_list = os.listdir(label_val_parent_path)
    # 获取文件路径名列表（父路径+文件名）
    label_val_file_path_list = []
    for name in label_val_name_list:
        label_val_file_path_list.append(os.path.join(label_val_parent_path + '/', name))
    # 依次打开label文件
    for file in label_val_file_path_list:
        # 打开文件，读
        with open(file, 'r', encoding='UTF-8') as f:
            val_labels = [x.split() for x in f.read().strip().splitlines()]  # 每组标签保存到一个维度（x，56）
            val_labels = np.array(val_labels, dtype=np.float32)  # 转为numpy数组
            fixed_val_labels = []  # 存放修改后的标签数据
            # 对每组标签，删除最后12个数据，即后4个关键点的数据
            for i in range(len(val_labels)):
                fixed_val_label = np.delete(val_labels[i], np.arange(-13, -1))
                fixed_val_labels.append(fixed_val_label)  # 删除后的新数组追加到一起，构成一个新列表
            fixed_val_labels = np.array(fixed_val_labels, dtype=np.float32)  # 转为numpy数组
            print("文件路径：" + str(file) + "," + "修改前:" + str(val_labels.shape) + ","+ "修改后:" + str(fixed_val_labels.shape))
            f.close()
        # 打开文件，写
        with open(file, "w") as f:
            # 分别写入每一组标签
            for i in range(len(fixed_val_labels)):
                cls = True  # 标志当前数是否是类别位，类别位是整数，其他位保留6位小数
                for num in fixed_val_labels[i]:
                    if cls:
                        num = int(num)
                    elif not cls:
                        num = format(num, '.6f')
                    cls = False
                    f.write(str(num) + ' ')  # 每个数字用空格隔开
                f.write('\n')  # 每组标签用换行符隔开
            f.close()


if __name__ == '__main__':
     fix_label_val()
