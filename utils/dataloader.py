import random

import cv2
import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader
from torch.utils.data.dataset import Dataset
from utils.util import cvtColor, preprocess_input, resize_image


class FaceNetDataset(Dataset):
    def __init__(self, input_shape, lines, random):
        self.num_classes = 0
        self.input_shape = input_shape
        self.lines = lines
        self.random = random

        # ------------------------------------#
        #   路径和标签
        # ------------------------------------#
        self.paths = []
        self.labels = []

        self.load_dataset()

    def __getitem__(self, item):
        # 创建一个3, 3,inputshape[0] inputshape[1] 的空tensor
        images = np.zeros((3, 3, self.input_shape[0], self.input_shape[1]))
        labels = np.zeros((3))

        #  先获得两张同一个人的人脸
        #  用来作为anchor和positive

        # 随机选择一个人脸标签
        random_index = random.randint(0, self.num_classes - 1)
        # 取出这个标签所有的人脸图
        img_path = self.paths[self.labels == random_index]
        while len(img_path) < 2:
            random_index = random.randint(0, self.num_classes - 1)
            img_path = self.paths[self.labels == random_index]

        # 随机取2张这个标签的人脸图
        image_indexes = np.random.choice(range(0, len(img_path)), 2)
        image_anchor = cvtColor(Image.open(img_path[image_indexes[0]]))
        image_anchor = resize_image(image_anchor, [self.input_shape[1], self.input_shape[0]], True)
        # image_anchor.show()
        image_anchor = preprocess_input(np.array(image_anchor, dtype='float32'))
        image_anchor = np.transpose(image_anchor, [2, 0, 1])
        images[0, :, :, :] = image_anchor
        labels[0] = random_index

        # 取第二个人脸作为 positive
        image_positive = cvtColor(Image.open(img_path[image_indexes[1]]))
        image_positive = resize_image(image_positive,[self.input_shape[1], self.input_shape[0]], True)
        # image_anchor.show()
        image_positive = preprocess_input(np.array(image_positive, dtype='float32'))
        image_positive = np.transpose(image_positive, [2, 0, 1])
        images[1, :, :, :] = image_positive
        labels[1] = random_index

        # 取出另一个人的人脸作为negative
        exclude_random_list = list(range(self.num_classes))
        exclude_random_list.pop(random_index)
        index = np.random.choice(range(0, (self.num_classes - 1)), 1)
        current_c = exclude_random_list[index[0]]
        selected_path = self.paths[self.labels == current_c]
        while len(selected_path) < 1:
            index = np.random.choice(range(0, self.num_classes - 1), 1)
            current_c = exclude_random_list[index[0]]
            selected_path = self.paths[self.labels == current_c]

        image_indexes = np.random.choice(range(0, len(selected_path)), 1)
        image_negative = cvtColor(Image.open(selected_path[image_indexes[0]]))
        image_negative = resize_image(image_negative, [self.input_shape[1], self.input_shape[0]], True)
        image_negative = preprocess_input(np.array(image_negative, dtype='float32'))
        image_negative = np.transpose(image_negative, [2, 0, 1])
        images[2, :, :, :] = image_negative
        labels[2] = current_c

        return images, labels

    def __len__(self):
        return len(self.lines)

    def load_dataset(self):
        for line in self.lines:
            label, img_path = line.split(';')
            self.labels.append(int(label))
            self.paths.append(img_path.split()[0])

        self.labels = np.array(self.labels)
        self.paths = np.array(self.paths)
        self.num_classes = np.max(self.labels) + 1


def dataset_collate(batch):
    images = []
    labels = []
    for img, label in batch:
        images.append(img)
        labels.append(label)

    np_images = np.array(images)
    image_anchor = np_images[:, 0, :, :, :]
    image_positive = np_images[:, 1, :, :, :]
    image_negative = np_images[:, 2, :, :, :]
    images = np.concatenate((image_anchor, image_positive, image_negative), axis=0)

    np_labels = np.array(labels)
    label_anchor = np_labels[:, 0]
    label_positive = np_labels[:, 1]
    label_negative = np_labels[:, 2]
    labels = np.concatenate((label_anchor, label_positive, label_negative), axis=0)

    images = torch.from_numpy(images).type(torch.FloatTensor)
    labels = torch.from_numpy(labels).type(torch.LongTensor)

    return images, labels

if __name__ == '__main__':
    annotation_path = "F:\\py\\face_net\\facenet-pytorch\\cls_train.txt"
    num_classes = 0
    paths = []
    labels = []
    with open(annotation_path, "r") as f:
        lines = f.readlines()
        for line in lines:
            label, img_path = line.split(';')
            labels.append(label)
            paths.append(img_path)

    np.random.shuffle(lines)

    dataset = FaceNetDataset((160, 160), lines, False)
    gen = DataLoader(dataset, shuffle=True, batch_size=96 // 3, drop_last=True, collate_fn=dataset_collate)
    for iteration, batch in enumerate(gen):
        pass
