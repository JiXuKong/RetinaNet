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
from data_pre.crop_image_label import txt_2_xml

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
    img_path = r'F:\open_dataset\90percentunlabel\JPEGImages'
    saved_img = r'F:\open_dataset\90percentunlabel\pred'
    saved_xml = r'F:\open_dataset\90percentunlabel\Annotations'
#     img_path = r'E:\python_files\luoshuan_detection\test\JPEGImages'
#     saved_img = r'E:\python_files\luoshuan_detection\test\predimg'
#     saved_xml = r'E:\python_files\luoshuan_detection\test\predlabel'
    
    total_timer = Timer()

    pred_classification_target_list, pred_regress_target_list = net.pred_classification_target_list, net.pred_regress_target_list
    
    pred_classification_target_list = tf.reshape(pred_classification_target_list, [-1, cfg.class_num-1])
    pred_classification_target_list = tf.nn.sigmoid(pred_classification_target_list)
    pred_regress_target_list = tf.reshape(pred_regress_target_list, [-1, 4])

    pred_regress_target_list = reverse_regress_target_tf(pred_regress_target_list, net.anchor)
    #

    
    pred_regress_i = pred_regress_target_list
    pred_classification_i = pred_classification_target_list
    nms_box, nms_score, nms_label = gpu_nms(pred_regress_i, pred_classification_i, cfg.class_num-1, 100, 0.9, 0.45)

    
    
    for fil in os.listdir(img_path):
        imgnm = os.path.join(img_path, fil)

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


        b, s, l = sess.run([nms_box, nms_score, nms_label], feed_dict = feed_dict)
        pred_b = b
        pred_s = s
        pred_l = l

        saved_box = []
        for j in range(pred_b.shape[0]):
            if (pred_l[j]>=0):
                
                x1,y1, x2, y2 = pred_b[j][0]*resize_scale_x, pred_b[j][1]*resize_scale_y, pred_b[j][2]*resize_scale_x, pred_b[j][3]*resize_scale_y
                saved_box.append([cfg.classes[pred_l[j]+1], int(x1),int(y1), int(x2),int(y2)])
                cv2.rectangle(img_orig,(int(x1),int(y1)),(int(x2),int(y2)),(0,255,0),2)
                cv2.putText(img_orig, cfg.classes[pred_l[j]+1] + str(pred_s[j]),(int(x1),int(y1)),cv2.FONT_HERSHEY_PLAIN,2.0,(255,255,255),2,1)
            
            cv2.imwrite(os.path.join(saved_img, fil), img_orig)
                
            imageList = {
                        'pose':'Unspecified',
                        'truncated':0,
                        'difficult':0,
                        'img_w':int(x2)-int(x1),
                        'img_h':int(y2)-int(y1),
                        'image_path':imgnm,
                        'img_name':fil.split('.')[0],
                        'object':saved_box
                                }
            txt_2_xml(imageList, saved_xml, fil.split('.')[0])
            
#         cv2.imshow("x",img_orig)
#         cv2.waitKey(0)
    
