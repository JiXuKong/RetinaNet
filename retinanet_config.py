import numpy as np

#some file path
# ckecpoint_file = './luosuanretinanet'
# train_img_txt = r'F:\bolt1\cut\train\train.txt'
# train_img_path = r'F:\bolt1\cut\train\img'
# train_label_path = r'F:\bolt1\cut\train\xml'

# test_img_txt = r'F:\bolt1\cut\test\test.txt'
# test_img_path = r'F:\bolt1\cut\test\img'
# test_label_path = r'F:\bolt1\cut\test\xml'

# ssl_img_txt = r'F:\bolt1\cut\test\test.txt'
# ssl_img_path = r'F:\bolt1\cut\test\img'
# ssl_label_path = r'F:\bolt1\cut\test\xml'


ckecpoint_file = './retinanet'
train_img_txt = r'F:\open_dataset\10percentlabel\labeled.txt'
train_img_path = r'F:\open_dataset\10percentlabel\JPEGImages'
train_label_path = r'F:\open_dataset\10percentlabel\Annotations'

# train_img_txt = r'F:\open_dataset\voc07+12\VOCdevkit\train\train.txt'
# train_img_path = r'F:\open_dataset\voc07+12\VOCdevkit\train\JPEGImages'
# train_label_path = r'F:\open_dataset\voc07+12\VOCdevkit\train\Annotations'

test_img_txt = r'F:\open_dataset\voc07+12\VOCdevkit\test\test.txt'
test_img_path = r'F:\open_dataset\voc07+12\VOCdevkit\test\JPEGImages'
test_label_path = r'F:\open_dataset\voc07+12\VOCdevkit\test\Annotations'

ssl_img_txt = r'F:\open_dataset\90percentunlabel\ssl_label.txt'
ssl_img_path = r'F:\open_dataset\90percentunlabel\JPEGImages'
ssl_label_path = r'F:\open_dataset\90percentunlabel\Annotations'




cache_path = './pkl1'
# cache_path = r'E:\python_files\luoshuan_detection\pkl'
# val_restore_path =  './luosuanretinanet/model.ckpt-16000'
# val_restore_path =  './retinanet/model.ckpt-150000'
# val_restore_path = r'E:\python_files\object_detection\luosuanretinanet\model.ckpt-18000'
# train_restore_path =  './pretrain_weight/ssd_resnet50_v1_fpn/model.ckpt'
train_restore_path = './pretrain_weight/resnet_v1_50_2016_08_28/resnet_v1_50.ckpt'
# train_restore_path = './retinanet/model.ckpt-90000'
# val_restore_list = ['./luosuanretinanet/model.ckpt-' + str(i) for i in range(5000, 20000, 3000)]


#data parameter
classes = ['background', 'aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus',
           'car', 'cat', 'chair', 'cow', 'diningtable', 'dog', 'horse',
           'motorbike', 'person', 'pottedplant', 'sheep', 'sofa',
           'train', 'tvmonitor']
# classes = ['background', 'bolt']
gridmask = False
random_crop = False
other_aug = False


#some network parameters
weight_decay = 0.0001
gradient_clip_by_norm = 10
batch_size = 1#测试时改为1
max_epoch = 1
train_num = 16551
test_num = 4952//batch_size
# test_num = 752//batch_size
class_num = len(classes)
image_size = 640#currently only 640
feature_size = [[image_size//(2**i), image_size//(2**i)] for i in range(3,8)]
phase = True
base_anchor = [32, 64, 128, 256, 512]
scale = np.array([1, 2**(1/2)])
aspect_ratio = np.array([1.0, 2.0, 0.5])
anchors = scale.shape[0]*aspect_ratio.shape[0]
momentum_rate = 0.9
alpha = 0.25
gama = 2
class_weight = 1.0
regress_weight = 1.0
decay = 0.99
pi = 1e-2