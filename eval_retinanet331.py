import tensorflow as tf
from tensorflow.python import pywrap_tensorflow
from retinanet331 import Retinanet
from part.pascal_voc import pascal_voc
from part.timer import Timer
import retinanet_config as cfg
from learning_schedules import cosine_decay_with_warmup
from part.NMS import cpu_nms, gpu_nms
from part.regress_target import reverse_regress_target_tf
import os
import sys
from evalue import voc_eval
import numpy as np
import math
import tensorflow.contrib.slim as slim

MOVING_AVERAGE_DECAY = 0.993
net = Retinanet(False)
val_data = pascal_voc(phase='val', flipped=False, img_path = cfg.test_img_path,\
                  label_path=cfg.test_label_path, img_txt=cfg.test_img_txt, is_training=False)
ckecpoint_file = cfg.ckecpoint_file
restore_path = cfg.val_restore_path

max_epoch = cfg.max_epoch
if not os.path.exists(ckecpoint_file):
    os.makedirs(ckecpoint_file)



sess = tf.Session()
sess.run(tf.global_variables_initializer())
summary_writer = tf.summary.FileWriter(ckecpoint_file, sess.graph)



if restore_path is not None:
    print('Restoring weights from: ' + restore_path)
    restorer = tf.train.Saver()
    restorer.restore(sess, restore_path)
    
if __name__ == '__main__':

    val_timer = Timer()

    epoch = 1

    pred_classification_target_list, pred_regress_target_list = tf.nn.sigmoid(net.pred_classification_target_list), net.pred_regress_target_list
    pred_classification_target_list, pred_regress_target_list = tf.stop_gradient(pred_classification_target_list), tf.stop_gradient(pred_regress_target_list)
    pred_classification_target_list = tf.reshape(pred_classification_target_list, [cfg.batch_size, -1, cfg.class_num-1])

    pred_regress_target_list = tf.reshape(pred_regress_target_list, [cfg.batch_size, -1, 4])
    boxes = []
    scores = []
    label = []
    for i in range(cfg.batch_size):
        pred_classification_target_i = pred_classification_target_list[i]
        is_confident = tf.reduce_max(pred_classification_target_i, axis=1) >= 0.001  # shape [N]
        encoded_boxes = tf.boolean_mask(pred_regress_target_list[i], is_confident)
        pred_scores = tf.boolean_mask(pred_classification_target_i, is_confident)  # shape [num_confident, num_classes]
        chosen_anchors = tf.boolean_mask(net.anchor, is_confident)  # shape [num_confident, 4]

        pred_regress_box_i = reverse_regress_target_tf(encoded_boxes, chosen_anchors)
        nms_box, nms_score, nms_label = gpu_nms(pred_regress_box_i, pred_scores, cfg.class_num-1, 100, 0.001, 0.5)
        boxes.append(nms_box)
        scores.append(nms_score)
        label.append(nms_label)

    val_pred = []
    gt_dict = {}
    val_rloss = 0
    val_closs = 0
    val_clone_loss = 0
    for val_step in range(1, cfg.test_num+1):
        val_timer.tic()
        val_images, val_labels,val_imnm, val_num_boxes = val_data.get()
        val_feed_dict = {net.image: val_images,
                     net.label: val_labels,
                     net.num_boxes:val_num_boxes
                    }

        b, s, l, valrloss_, valcloss_, valtotal_loss = sess.run([boxes, scores, label, net.loc_loss, net.cls_loss, net.total_loss],
                                                      feed_dict = val_feed_dict)
        val_rloss += valrloss_/cfg.test_num
        val_closs += valcloss_/cfg.test_num
        val_clone_loss += (valrloss_ + valcloss_)/cfg.test_num

        for i in range(cfg.batch_size):
            pred_b = b[i]
            pred_s = s[i]
            pred_l = l[i]
            for j in range(pred_b.shape[0]):
                if pred_l[j] >=0 :
                    val_pred.append([val_imnm[i], pred_b[j][0], pred_b[j][1], pred_b[j][2], pred_b[j][3], pred_s[j], pred_l[j]+1])
            single_gt_num = np.where(val_labels[i][:,0]>0)[0].shape[0]
            box = np.hstack((val_labels[i][:single_gt_num, 1:], np.reshape(val_labels[i][:single_gt_num, 0], (-1,1)))).tolist()
            gt_dict[val_imnm[i]] = box 

        val_timer.toc()    
        sys.stdout.write('\r>> ' + 'val_nums '+str(val_step)+str('/')+str(cfg.test_num+1))
        sys.stdout.flush()

    print('curent val speed: ', val_timer.average_time, 'val remain time: ', val_timer.remain(val_step, cfg.test_num+1))
    print('val mean regress loss: ', val_rloss, 'val mean class loss: ', val_closs, 'val mean total loss: ', val_clone_loss)
    mean_rec = 0
    mean_prec = 0
    mAP = 0
    for classidx in range(1, cfg.class_num):#从1到21，对应[bg,...]21个类（除bg）
        rec, prec, ap = voc_eval(gt_dict, val_pred, classidx, iou_thres=0.5, use_07_metric=False)
        print(cfg.classes[classidx] + ' ap: ', ap)
        mean_rec += rec[-1]/(cfg.class_num-1)
        mean_prec += prec[-1]/(cfg.class_num-1)
        mAP += ap/(cfg.class_num-1)

    val_total_summary2 = tf.Summary(value=[
        tf.Summary.Value(tag="val/loss/class_loss", simple_value=val_closs),
        tf.Summary.Value(tag="val/loss/regress_loss", simple_value=val_rloss),
        tf.Summary.Value(tag="val/loss/clone_loss", simple_value=val_clone_loss),
        tf.Summary.Value(tag="val/mA", simple_value=mAP),
        tf.Summary.Value(tag="val/mRecall", simple_value=mean_rec),
        tf.Summary.Value(tag="val/mPrecision", simple_value=mean_prec),
     ])
    summary_writer.add_summary(val_total_summary2, epoch)
    print('Epoch: ' + str(epoch), 'mAP: ', mAP)
    print('Epoch: ' + str(epoch), 'mRecall: ', mean_rec)
    print('Epoch: ' + str(epoch), 'mPrecision: ', mean_prec)