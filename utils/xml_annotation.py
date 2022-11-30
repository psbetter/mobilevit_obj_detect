import argparse
import glob
import os
import random
import xml.etree.ElementTree as ET

import numpy as np


def get_class_names(root):
    class_names = []
    xmls = glob.glob(f"{root}/*/*.xml")
    for xml_path in xmls:
        objects = ET.parse(xml_path.rstrip()).findall("object")
        for object in objects:
            class_name = object.find('name').text
            if class_name not in class_names:
                class_names.append(class_name)
    return class_names

        
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train classification of CMT model")
    parser.add_argument('--save_path', default='../config', type=str, help='save path')
    parser.add_argument('--root_path', default="E:/PyCharmWorkSpace/datasets/mask", type=str, help='dataset root path')
    args = parser.parse_args()
    random.seed(0)

    print("[INFO] Generate train.txt and val.txt for train.")
    class_names = get_class_names(args.root_path)
    class_dict = {class_name: i for i, class_name in enumerate(class_names)}

    with open(f"{args.save_path}/class_names.txt", 'w') as f:
        for class_name in class_names:
            f.write(class_name + '\n')

    total_xmls = glob.glob(f"{args.root_path}/train/*.xml")
    total_xmls = np.array(total_xmls)
    total_size = len(total_xmls)
    train_size = int(0.8 * total_size)
    val_size = total_size - train_size
    print("[INFO] train size: {}, val size: {}".format(train_size, val_size))
    shuffled_idx = np.arange(total_size)
    np.random.shuffle(shuffled_idx)
    train_xmls = total_xmls[shuffled_idx[:train_size]]
    val_xmls = total_xmls[shuffled_idx[train_size:]]

    with open(f"{args.save_path}/train.txt", 'w') as path:
        for xml_path in train_xmls:
            img_path = xml_path[:-3] + 'jpg'
            path.write(img_path)
            objects = ET.parse(xml_path).findall("object")
            boxes = []
            for object in objects:
                class_name = object.find('name').text
                bbox = object.find('bndbox')
                x1 = float(bbox.find('xmin').text) - 1
                y1 = float(bbox.find('ymin').text) - 1
                x2 = float(bbox.find('xmax').text) - 1
                y2 = float(bbox.find('ymax').text) - 1
                boxes.append([x1, y1, x2, y2])
                box_info = " %d,%d,%d,%d,%d" % (x1, y1, x2, y2, class_dict[class_name])
                path.write(box_info)
            path.write("\n")

    with open(f"{args.save_path}/val.txt", 'w') as path:
        for xml_path in val_xmls:
            img_path = xml_path[:-3] + 'jpg'
            path.write(img_path)
            objects = ET.parse(xml_path).findall("object")
            boxes = []
            for object in objects:
                class_name = object.find('name').text
                bbox = object.find('bndbox')
                x1 = float(bbox.find('xmin').text) - 1
                y1 = float(bbox.find('ymin').text) - 1
                x2 = float(bbox.find('xmax').text) - 1
                y2 = float(bbox.find('ymax').text) - 1
                boxes.append([x1, y1, x2, y2])
                box_info = " %d,%d,%d,%d,%d" % (x1, y1, x2, y2, class_dict[class_name])
                path.write(box_info)
            path.write("\n")

    print('[INFO] Generate data path txt finished!')
