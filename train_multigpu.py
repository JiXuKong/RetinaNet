import tensorflow as tf
from tensorflow.python import pywrap_tensorflow
from multigpuretinanet import Retinanet
from part.pascal_voc import pascal_voc
from part.tf_decode import get_generator
from part.timer import Timer
import retinanet_config as cfg
from part.NMS import cpu_nms, gpu_nms
from part.regress_target import reverse_regress_target_tf
from learning_schedules import cosine_decay_with_warmup
import os
import sys
import retinanet_config as cfg
from evalue import voc_eval
import numpy as np
import math
import tensorflow.contrib.slim as slim

MOVING_AVERAGE_DECAY = 0.993
init_lr = 1e-3
os.environ["CUDA_VISIBLE_DEVICES"] = '0,1'
data = pascal_voc('train')
val_data = pascal_voc('val')
ckecpoint_file = cfg.ckecpoint_file
restore_path = cfg.restore_path
flipped = cfg.flipped
train_num = cfg.train_num
batch_size = cfg.batch_size
epoch_step = train_num*(1 + int(flipped))//batch_size
max_epoch = cfg.max_epoch
if not os.path.exists(ckecpoint_file):
    os.makedirs(ckecpoint_file)
global_step = tf.train.create_global_step()
with tf.variable_scope('learning_rate'):
#     global_step = tf.train.get_global_step()
#     global_step = tf.placeholder(tf.int32, , shape = [])
    learning_rate = cosine_decay_with_warmup(
        global_step = global_step,
        learning_rate_base = 0.0043333/32,
        total_steps = int(3e5),
        warmup_learning_rate=0.0013333/32,
        warmup_steps=int(1e4))
momentum_rate = cfg.momentum_rate

gpu = ['0','1']


    

def average_gradients(tower_grads):
    """Calculate the average gradient for each shared variable across all towers.
    Note that this function provides a synchronization point across all towers.
    Args:
      tower_grads: List of lists of (gradient, variable) tuples. The outer list
        is over individual gradients. The inner list is over the gradient
        calculation for each tower.
    Returns:
       List of pairs of (gradient, variable) where the gradient has been averaged
       across all towers.
    """
    average_grads = []
    for grad_and_vars in zip(*tower_grads):
        # Note that each grad_and_vars looks like the following:
        #   ((grad0_gpu0, var0_gpu0), ... , (grad0_gpuN, var0_gpuN))
        grads = []
        for g, _ in grad_and_vars:
            # Add 0 dimension to the gradients to represent the tower.
            expanded_g = tf.expand_dims(g, 0)

            # Append on a 'tower' dimension which we will average over below.
            grads.append(expanded_g)

        # Average over the 'tower' dimension.
        grad = tf.concat(axis=0, values=grads)
        grad = tf.reduce_mean(grad, 0)

        # Keep in mind that the Variables are redundant because they are shared
        # across towers. So .. we will just return the first tower's pointer to
        # the Variable.
        v = grad_and_vars[0][1]
        grad_and_var = (grad, v)
        average_grads.append(grad_and_var)
    return average_grads
    
    
def get_variables_in_checkpoint_file(file_name):
    reader = pywrap_tensorflow.NewCheckpointReader(file_name)
    var_to_shape_map = reader.get_variable_to_shape_map()
    return var_to_shape_map

def initialize(pretrained_model, variable_to_restore):
    var_keep_dic = get_variables_in_checkpoint_file(pretrained_model)
    # Get the variables to restore, ignoring the variables to fix
    variables_to_restore = get_variables_to_restore(variable_to_restore, var_keep_dic)
    restorer = tf.train.Saver(variables_to_restore)
    return restorer

def get_variables_to_restore(variables, var_keep_dic):
    variables_to_restore = []

    for v in variables:
        if (v.name == 'global_step:0'):
            continue
#         print(v.name)
        if len(v.name.split('/')) > 1:
            if (v.name.split('/')[-1] != 'Momentum')\
            and(v.name.split(':')[0])in var_keep_dic:
                print('Variables restored: %s' % v.name)
                variables_to_restore.append(v)
        
    return variables_to_restore


update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
with tf.control_dependencies(update_ops):
    optimizer = tf.train.MomentumOptimizer(learning_rate, momentum_rate, use_nesterov=True)

with tf.variable_scope(tf.get_variable_scope()):
    print(len(gpu))
    p = 0
    add = cfg.batch_size//len(gpu)
    sum_total_loss, sum_l2, sum_rl, sum_cl, sum_p_nm = 0, 0, 0, 0, 0
    tower_grads = []
#     pred_score = []
#     pred_delta_box = []
    with tf.device('cpu:0'):
        inpu_image = tf.placeholder(tf.float32, shape = [None, cfg.image_size, cfg.image_size, 3])
        inpu_label = tf.placeholder(tf.int32, shape = [None, None, 5])
        input_num_boxes = tf.placeholder(tf.int32, shape = [None])
#         inpu_anchor = tf.placeholder(tf.float32, shape = [None, 4])
    for i in range(len(gpu)):
        print(i)
        with tf.device('gpu:%d' % i):
            with tf.name_scope('tower_%d' % i):
                with slim.arg_scope(
                                [slim.model_variable, slim.variable],
                                device='/device:CPU:0'):
                    label_i = inpu_label[i*add:(i+1)*add]
                    image_i = inpu_image[i*add:(i+1)*add]
                    num_boxes_i = input_num_boxes[i*add:(i+1)*add]
#                     anchor_i = inpu_anchor
                    net = Retinanet(image_i, label_i, num_boxes_i)
#                     pred_score.append(net.pred_score_target_list)
#                     pred_delta_box.append(net.pred_regress_target_list)
                    sum_total_loss += net.total_loss/len(gpu)
                    sum_l2 += net.regular_loss/len(gpu)
                    sum_rl += net.loc_loss/len(gpu)
                    sum_cl += net.cls_loss/len(gpu)
                    sum_p_nm += net.normalizer/len(gpu)
#                     optimizer = tf.train.MomentumOptimizer(learning_rate, momentum_rate, use_nesterov=True)
            tf.get_variable_scope().reuse_variables()
            with tf.control_dependencies(update_ops):
                gradient_i = optimizer.compute_gradients(net.total_loss)
                tower_grads.append(gradient_i)
                

if len(tower_grads) > 1:
    gradient = average_gradients(tower_grads)
else:
    gradient = tower_grads[0]
with tf.device('cpu:0'):
    pred_score = tf.concat(pred_score, axis = 0)
    pred_delta_box = tf.concat(pred_delta_box, axis = 0)
    with tf.control_dependencies(update_ops):
        with tf.name_scope('clip_gradients_YJR'):
            gradient = slim.learning.clip_gradient_norms(gradient, cfg.gradient_clip_by_norm)
            train_op = optimizer.apply_gradients(gradient,global_step=global_step)   

# with tf.control_dependencies([train_op]), tf.name_scope('ema'):
#     ema = tf.train.ExponentialMovingAverage(decay=MOVING_AVERAGE_DECAY, num_updates=global_step)
#     train_op = ema.apply(tf.trainable_variables()) 
    
g_list = tf.global_variables()
save_list = [g for g in g_list if ('Momentum' not in g.name)and('ExponentialMovingAverage' not in g.name)]
for v in save_list:
    print(v.name)
saver = tf.train.Saver(var_list=save_list, max_to_keep=20)
sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True))
sess.run(tf.group(tf.global_variables_initializer(),tf.local_variables_initializer()))
summary_writer = tf.summary.FileWriter(ckecpoint_file, sess.graph)

    
if restore_path is not None:
    print('Restoring weights from: ' + restore_path)
#     restorer = tf.train.Saver()
    restorer = initialize(restore_path, g_list)
    restorer.restore(sess, restore_path)
    
if __name__ == '__main__':

    total_timer = Timer()
    train_timer = Timer()
    load_timer = Timer()
    val_timer = Timer()
    if cfg.data_use_tfrecords:
        train_gen = get_generator(cfg.train_tfrecord_path, cfg.batch_size, cfg.buffer_size)
        init_op, iterator = train_gen
        sess.run(init_op)
                
#         train_img, train_label = load_data(cfg.train_tfrecord_path)
#         train_img_batch, train_label_batch = tf.train.shuffle_batch([train_img,train_label],batch_size=cfg.batch_size,num_threads=64, capacity=1012,min_after_dequeue=1000)
#         coord=tf.train.Coordinator()
#         threads = tf.train.start_queue_runners(sess, coord)
    anchorlist = net.anchorlist
    t = 1

    for epoch in range(1, max_epoch + 1):
        print('-'*25, 'epoch', epoch,'/',str(max_epoch), '-'*25)

        t_loss = 0
        ll_loss = 0
        r_loss = 0
        c_loss = 0
        
        
        for step in range(1, epoch_step + 1):
#             if t<=2000:
#                 lr = 1e-4*t/200
#             if 2000<t<=65000:
#                 lr = 1e-3
#             if 65000<t<=75000:
#                 lr = 1e-4
#             if t>75000:
#                 lr = 1e-5

            t = t + 1
            total_timer.tic()
            load_timer.tic()
            if cfg.data_use_tfrecords:
                img, l = iterator.get_next()
                images, labels = sess.run([img, l])
#                 images, labels = sess.run([train_img_batch, train_label_batch])
            else:
                images, labels, imnm, num_boxes = data.get()
            
            load_timer.toc()

            feed_dict = {inpu_image: images,
                         inpu_label: labels,
                         input_num_boxes: num_boxes
#                          net.is_training:True
                        }
            _, g_step_, total_loss, l2_loss, rloss_, closs_, p_nm_, lr = sess.run(
            [train_op,
             global_step,
             sum_total_loss,
             sum_l2, 
             sum_rl, 
             sum_cl,
             sum_p_nm, learning_rate], feed_dict = feed_dict)
             
            if step%50 ==0:
                sys.stdout.write('\r>> ' + 'iters '+str(step)+str('/')+str(epoch_step)+' loss '+str(total_loss) + ' ')
                sys.stdout.flush()

                train_total_summary = tf.Summary(value=[
                    tf.Summary.Value(tag="config/learning rate", simple_value=lr),
                    tf.Summary.Value(tag="train/classification/focal_loss", simple_value=cfg.class_weight*closs_),
                    tf.Summary.Value(tag="train/p_nm", simple_value=p_nm_),
                    tf.Summary.Value(tag="train/regress_loss", simple_value=cfg.regress_weight*rloss_),
                    tf.Summary.Value(tag="train/clone_loss", simple_value=cfg.class_weight*closs_ + cfg.regress_weight*rloss_),
                    tf.Summary.Value(tag="train/l2_loss", simple_value=l2_loss),
                    tf.Summary.Value(tag="train/total_loss", simple_value=total_loss)
                    ])
                summary_writer.add_summary(train_total_summary, t)
        total_timer.toc()
        sys.stdout.write('\n')
        print('>> mean loss', t_loss)
        print('curent speed: ', total_timer.average_time, 'remain time: ', total_timer.remain(g_step_, epoch_step*max_epoch))

        print('saving checkpoint')
        saver.save(sess, ckecpoint_file + '/model.ckpt', epoch)

        #val
        if epoch%cfg.val_epoch == 0:
            pred_classification_target_list, pred_regress_target_list = pred_score, pred_delta_box
            pred_classification_target_list = tf.reshape(pred_classification_target_list, [batch_size, -1, cfg.class_num-1])
            pred_regress_target_list = tf.reshape(pred_regress_target_list, [batch_size, -1, 4])
            boxes = []
            scores = []
            label = []
            for i in range(cfg.batch_size):
                pred_classification_target_i = pred_classification_target_list[i]
                is_confident = tf.reduce_max(pred_classification_target_i, axis=1) >= 0.01  # shape [N]
                encoded_boxes = tf.boolean_mask(pred_regress_target_list[i], is_confident)
                pred_scores = tf.boolean_mask(pred_classification_target_i, is_confident)  # shape [num_confident, num_classes]
                chosen_anchors = tf.boolean_mask(net.anchor, is_confident)  # shape [num_confident, 4]
                
                pred_regress_box_i = reverse_regress_target_tf(encoded_boxes, chosen_anchors)
                nms_box, nms_score, nms_label = gpu_nms(pred_regress_box_i, pred_scores, cfg.class_num-1, 50, 0.01, 0.5)
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
                val_images, val_labels,imnm = val_data.get()
                val_feed_dict = {inpu_image: val_images,
                             inpu_label: val_labels
                            }
                
                b, s, l, valrloss_, valcloss_, valtotal_loss = sess.run([boxes, scores, label, sum_rl, sum_cl, sum_total_loss],
                                                              feed_dict = val_feed_dict)
                val_rloss += valrloss_/cfg.test_num
                val_closs += valcloss_/cfg.test_num
                val_clone_loss += (valrloss_ + valcloss_)/cfg.test_num

                for i in range(batch_size):
                    pred_b = b[i]
                    pred_s = s[i]
                    pred_l = l[i]
                    for j in range(pred_b.shape[0]):
                        if pred_l[j] >=0 :
                            val_pred.append([imnm[i], pred_b[j][0], pred_b[j][1], pred_b[j][2], pred_b[j][3], pred_s[j], pred_l[j]+1])
                    single_gt_num = np.where(val_labels[i][:,0]>0)[0].shape[0]
                    box = np.hstack((val_labels[i][:single_gt_num, 1:], np.reshape(val_labels[i][:single_gt_num, 0], (-1,1)))).tolist()
                    gt_dict[imnm[i]] = box 

                val_timer.toc()    
                sys.stdout.write('\r>> ' + 'val_nums '+str(val_step)+str('/')+str(cfg.test_num+1))
                sys.stdout.flush()
#                 sys.stdout.write('\n')
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
#                 val_total_summary1 = tf.Summary(value=[
#                     tf.Summary.Value(tag="val/recall@iou0.5 " + cfg.classes[classidx], simple_value=rec[-1]),
#                     tf.Summary.Value(tag="val/precision/@iou0.5 " + cfg.classes[classidx], simple_value=prec[-1]),
#                     tf.Summary.Value(tag="val/AP@iou0.5 " + cfg.classes[classidx], simple_value=ap)])
#                 summary_writer.add_summary(val_total_summary1, epoch)
            val_total_summary2 = tf.Summary(value=[
                tf.Summary.Value(tag="val/loss/class_loss", simple_value=val_closs),
                tf.Summary.Value(tag="val/loss/regress_loss", simple_value=val_rloss),
                tf.Summary.Value(tag="val/loss/clone_loss", simple_value=val_clone_loss),
                tf.Summary.Value(tag="val/mAP@iou0.5", simple_value=mAP),
                tf.Summary.Value(tag="val/mRecall@iou0.5", simple_value=mean_rec),
                tf.Summary.Value(tag="val/mPrecision@iou0.5", simple_value=mean_prec),
             ])
            summary_writer.add_summary(val_total_summary2, epoch)
            print('Epoch: ' + str(epoch), 'mAP@iou0.5: ', mAP)
            print('Epoch: ' + str(epoch), 'mRecall@iou0.5: ', mean_rec)
            print('Epoch: ' + str(epoch), 'mPrecision@iou0.5: ', mean_prec)