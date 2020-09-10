import tensorflow as tf
import numpy as np
from retinanet331 import Retinanet
from part.timer import Timer
import retinanet_config as cfg
from part.NMS import cpu_nms, gpu_nms
from part.regress_target import reverse_regress_target_tf, regress_target_tf
import copy
import cv2
import pickle
import os
import sys


net = Retinanet(False)

restore_path = cfg.val_restore_path
g_list = tf.global_variables()

sess = tf.Session()
sess.run(tf.global_variables_initializer())

if restore_path is not None:
    print('Restoring weights from: ' + restore_path)
    restorer = tf.train.Saver(g_list)
    restorer.restore(sess, restore_path)
    

    
if __name__ == '__main__':
    total_timer = Timer()

    pred_classification_target_list, pred_regress_target_list = net.pred_classification_target_list, net.pred_regress_target_list
    
    pred_classification_target_list = tf.reshape(pred_classification_target_list, [-1, cfg.class_num-1])
    pred_classification_target_list = tf.nn.sigmoid(pred_classification_target_list)
    pred_regress_target_list = tf.reshape(pred_regress_target_list, [-1, 4])

    pred_regress_target_list = reverse_regress_target_tf(pred_regress_target_list, net.anchor)
    #

    
    pred_regress_i = pred_regress_target_list
    pred_classification_i = pred_classification_target_list
    nms_box, nms_score, nms_label = gpu_nms(pred_regress_i, pred_classification_i, cfg.class_num-1, 100, 0.3, 0.5)

    
    
    imgnm = r'F:\open_dataset\voc07+12\VOCdevkit\test\JPEGImages\008134.jpg'
#     imgnm = r'E:\python_files\Faster-RCNN_Tensorflow-master\tools\demos\1.jpg'

    img = cv2.imread(imgnm)

    y, x = img.shape[0:2]

    resize_scale_x = x/cfg.image_size
    resize_scale_y = y/cfg.image_size
    img_orig = copy.deepcopy(img)

    img = cv2.resize(img,(cfg.image_size,cfg.image_size))
    img=img[:,:,::-1]
    img=img.astype(np.float32, copy=False)
    mean = np.array([123.68, 116.779, 103.979])
    mean = mean.reshape(1,1,3)
    img = img - mean
    img = np.reshape(img, (1, cfg.image_size, cfg.image_size, 3))
    feed_dict = {
        net.image: img
                }


    b, s, l, pred_classification_i,pred_regress_i = sess.run([nms_box, nms_score, nms_label, pred_classification_i,pred_regress_i], feed_dict = feed_dict)
    pred_b = b
    pred_s = s
    pred_l = l

    for j in range(pred_b.shape[0]):
        if (pred_l[j]>=0):
            print(pred_l[j], pred_s[j])
            x1,y1, x2, y2 = pred_b[j][0]*resize_scale_x, pred_b[j][1]*resize_scale_y, pred_b[j][2]*resize_scale_x, pred_b[j][3]*resize_scale_y
            cv2.rectangle(img_orig,(int(x1),int(y1)),(int(x2),int(y2)),(0,255,0),2)
            cv2.putText(img_orig, str(cfg.classes[pred_l[j]+1]) + str(pred_s[j])[:3],(int(x1),int(y1)),cv2.FONT_HERSHEY_PLAIN,1,(255,255,255),1,1)

    cv2.imshow("x",img_orig)
    cv2.waitKey(0)
    
