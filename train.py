import numpy as np
import torch
from torch import optim

from nets.facenet import Facenet
from nets.facenet_training import get_lr_scheduler, set_optimizer_lr,triple_loss
from utils.dataloader import FaceNetDataset, dataset_collate
from torch.utils.data import DataLoader

from utils.utils_fit import fit_one_epoch

backbone = 'mobilenet'

input_shape = [160, 160, 3]

pretrained = True
batch_size = 96

model_path = "model_data/facenet_mobilenet.pth"

optimizer_type = "adam"
momentum = 0.9
weight_decay = 0

Init_lr = 1e-3
Min_lr = Init_lr * 0.01

lr_decay_type = 'cos'
Epoch = 100

save_dir = 'logs'
save_period = 10


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 加载数据集相关内容
annotation_path = "F:\\py\\face_net\\facenet-pytorch\\cls_train.txt"
num_classes = 0
paths = []
labels = []
with open(annotation_path, "r") as f:
    lines = f.readlines()
    for line in lines:
        label, img_path = line.split(';')
        labels.append(int(label))
        paths.append(img_path.split()[0])

np.random.shuffle(lines)

labels = np.array(labels)
paths = np.array(paths)
num_classes = np.max(labels) + 1

# 创建模型
model = Facenet(backbone=backbone, num_classes=num_classes)

if model_path != '':
    load_key, no_load_key, temp_dict = [], [], {}
    model_dict = model.state_dict()   # 模型的空字典有key 无 value ，现在就是拿字段key比较，key 一样的，把value 填充进去
    pretrained_dict = torch.load(model_path, map_location=device)
    for key, value in pretrained_dict.items():
        if key in pretrained_dict.keys() and model_dict[key].shape == value.shape:
            temp_dict[key] = value
            load_key.append(key)
        else:
            no_load_key.append(key)
    model_dict.update(temp_dict)
    model.load_state_dict(model_dict)

    print("\nSuccessful Load Key:", str(load_key)[:500], "……\nSuccessful Load Key Num:", len(load_key))
    print("\nFail To Load Key:", str(no_load_key)[:500], "……\nFail To Load Key num:", len(no_load_key))
    print("\n\033[1;33;44m温馨提示，head部分没有载入是正常现象，Backbone部分没有载入是错误的。\033[0m")
model = model.to(device)
# 0.01验证，0.99用于训练
val_split = 0.01
with open(annotation_path, "r") as f:
    lines = f.readlines()
np.random.seed(10101)
np.random.shuffle(lines)
np.random.seed(None)
num_val = int(len(lines)*val_split)
num_train = len(lines) - num_val

if batch_size % 3 != 0:
    raise ValueError("Batch_size must be the multiple of 3.")

#-------------------------------------------------------------------#
#   判断当前batch_size，自适应调整学习率
#-------------------------------------------------------------------#
nbs = 64
lr_limit_max = 1e-3 if optimizer_type == 'adam' else 1e-1
lr_limit_min = 3e-4 if optimizer_type == 'adam' else 5e-4
Init_lr_fit = min(max(batch_size / nbs * Init_lr, lr_limit_min), lr_limit_max)
Min_lr_fit = min(max(batch_size / nbs * Min_lr, lr_limit_min * 1e-2), lr_limit_max * 1e-2)

# ---------------------------------------#
#   根据optimizer_type选择优化器
# ---------------------------------------#
optimizer = {
    'adam': optim.Adam(model.parameters(), Init_lr_fit, betas=(momentum, 0.999), weight_decay=weight_decay),
    'sgd': optim.SGD(model.parameters(), Init_lr_fit, momentum=momentum, nesterov=True, weight_decay=weight_decay)
}[optimizer_type]

# 获得学习率公式
lr_scheduler_func = get_lr_scheduler(lr_decay_type, Init_lr_fit, Min_lr_fit, Epoch)

# 加载数据集
train_dataset = FaceNetDataset(input_shape, lines[:num_train], True)
val_dataset = FaceNetDataset(input_shape, lines[num_train:], False)
gen = DataLoader(train_dataset, shuffle=True, batch_size=batch_size // 3, drop_last=True, collate_fn=dataset_collate)
gen_val = DataLoader(val_dataset, shuffle=True, batch_size=batch_size // 3, drop_last=True, collate_fn=dataset_collate)

loss = triple_loss()

for epoch in range(0, Epoch):
    set_optimizer_lr(optimizer, lr_scheduler_func, epoch)
    fit_one_epoch(model, loss, gen, gen_val, device, optimizer, batch_size // 3, epoch, Epoch,
                  num_train // batch_size, num_val // batch_size, save_dir, save_period)
