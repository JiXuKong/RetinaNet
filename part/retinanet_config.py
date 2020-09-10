import numpy as np
ckecpoint_file = './retinanet'
# img_txt = r'F:\open_dataset\voc07+12\VOCdevkit\train\train.txt'
# img_path = r'F:\open_dataset\voc07+12\VOCdevkit\train\JPEGImages'
# label_path = r'F:\open_dataset\voc07+12\VOCdevkit\train\Annotations'

img_txt = r'E:\python_files\luoshuan_detection\train_less\train_less.txt'
img_path = r'E:\python_files\luoshuan_detection\train_less\JPEGImages'
label_path = r'E:\python_files\luoshuan_detection\train_less\Annotations'

train_tfrecord_path = [r'F:\back_up\object_detection\tfrecords1\train1.tfrecords',
                       r'F:\back_up\object_detection\tfrecords1\train2.tfrecords',
                       r'F:\back_up\object_detection\tfrecords1\train3.tfrecords',
                       r'F:\back_up\object_detection\tfrecords1\train4.tfrecords',
                       r'F:\back_up\object_detection\tfrecords1\train5.tfrecords']
# cache_path = '/pkl/pkl'
cache_path = r'E:\python_files\luoshuan_detection\pkl'
is_training = True
train = 'train'
# classes = ['background', 'aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus',
#            'car', 'cat', 'chair', 'cow', 'diningtable', 'dog', 'horse',
#            'motorbike', 'person', 'pottedplant', 'sheep', 'sofa',
#            'train', 'tvmonitor']
classes = ['background', 'bolt']

# restore_path = r'E:\python_files\object_detection\net\resnet_v1_50_2016_08_28\resnet_v1_50.ckpt'
restore_path =  r'E:\python_files\object_detection\retinanet\1\model.ckpt-7'
# restore_path =  r'E:\python_files\tensorflow_master\object_detection_API\OFFICIAL\models-master\models-master\research\object_detection\luoshuan_semi_supervised\train_ssd600_fpn_r50\fine_tune\ssd_resnet50_v1_fpn\model.ckpt'
weight_decay = 0.0001
gradient_clip_by_norm = 10
data_use_tfrecords = False
flipped = True
batch_size = 4
buffer_size = 512
train_num = 16551
# test_num = 20
test_num = 4952//batch_size
max_epoch = 16
class_num = 21
image_size = 640
feature_size = [[image_size//(2**i), image_size//(2**i)] for i in range(3,8)]
filter_list = [3, 4, 6, 3]
_bottleneck = True
phase = True
base_anchor = [32, 64, 128, 256, 512]
scale = np.array([1, 2**(1/2)])
# aspect_ratio = np.array([2,1/2,1])
aspect_ratio = np.array([1.0, 2.0, 0.5])
anchors = scale.shape[0]*aspect_ratio.shape[0]
momentum_rate = 0.9
alpha = 0.25
gama = 2
class_weight = 1.0
regress_weight = 1.0
l2_weight = 1e-4
decay = 0.99
pi = 1e-2
use_select_anchor_py = False
focal_loss = False
val_epoch = 1