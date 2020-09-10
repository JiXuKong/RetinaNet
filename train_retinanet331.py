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
import retinanet_config as cfg
from evalue import voc_eval
import numpy as np
import math
import tensorflow.contrib.slim as slim

MOVING_AVERAGE_DECAY = 0.993
init_lr = 1e-4
net = Retinanet(True)
data = pascal_voc(phase='train_full', flipped=True, img_path = cfg.train_img_path,\
                  label_path=cfg.train_label_path, img_txt=cfg.train_img_txt, is_training=True)
ckecpoint_file = cfg.ckecpoint_file
restore_path = cfg.train_restore_path
flipped = True
train_num = cfg.train_num
batch_size = cfg.batch_size
epoch_step = train_num*(1 + int(flipped))//batch_size
max_epoch = cfg.max_epoch
if not os.path.exists(ckecpoint_file):
    os.makedirs(ckecpoint_file)
global_step = tf.train.create_global_step()
with tf.variable_scope('learning_rate'):
    learning_rate = cosine_decay_with_warmup(
        global_step = global_step,
        learning_rate_base = 0.004333/32,
        total_steps = int(3.5e5),
        warmup_learning_rate=0.0013333/32,
        warmup_steps=int(2000))

momentum_rate = cfg.momentum_rate
update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
with tf.control_dependencies(update_ops):
    optimizer = tf.train.MomentumOptimizer(learning_rate, momentum_rate, use_nesterov=False)
    gradient = optimizer.compute_gradients(net.total_loss)
    
    
#     with tf.name_scope('clip_gradients_YJR'):
#         gradient = slim.learning.clip_gradient_norms(gradient,cfg.gradient_clip_by_norm)


    with tf.name_scope('apply_gradients'):
        train_op = optimizer.apply_gradients(grads_and_vars=gradient,global_step=global_step)
        


g_list = tf.global_variables()
save_list = [g for g in g_list if ('Momentum' not in g.name)and('ExponentialMovingAverage' not in g.name)]
saver = tf.train.Saver(var_list=save_list, max_to_keep=30)

    

sess = tf.Session()
sess.run(tf.global_variables_initializer())
summary_writer = tf.summary.FileWriter(ckecpoint_file, sess.graph)

    

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
            continue;

        if(v.name.split('/')[1] != 'ClassPredictor')\
        and(v.name.split('/')[1] != 'BoxPredictor')\
        and(v.name.split(':')[0])in var_keep_dic:

            print('Variables restored: %s' % v.name)
            variables_to_restore.append(v)
        
    return variables_to_restore



if restore_path is not None:
    print('Restoring weights from: ' + restore_path)
    restorer = initialize(restore_path, g_list)
#     restorer = tf.train.Saver(save_list)
    restorer.restore(sess, restore_path)
    
if __name__ == '__main__':

    total_timer = Timer()
    train_timer = Timer()
    load_timer = Timer()
    val_timer = Timer()

    max_epoch = 35
    t = 1
    for epoch in range(1, max_epoch + 1):
        print('-'*25, 'epoch', epoch,'/',str(max_epoch), '-'*25)


        t_loss = 0
        ll_loss = 0
        r_loss = 0
        c_loss = 0
        
        
       
        for step in range(1, epoch_step + 1):
     
            t = t + 1
            total_timer.tic()
            load_timer.tic()
 
            images, labels, imnm, num_boxes = data.get()
            
            load_timer.toc()
            feed_dict = {net.image: images,
                         net.label: labels,
                         net.num_boxes:num_boxes
                        }

            _, g_step_, total_loss, l2_loss, rloss_, closs_, p_nm_, lr = sess.run(
                [train_op,
                 global_step,
                 net.total_loss,
                 net.regular_loss, 
                 net.loc_loss, 
                 net.cls_loss,
                 net.normalizer,learning_rate], feed_dict = feed_dict)
            
            
            
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
                summary_writer.add_summary(train_total_summary, g_step_)
            if g_step_%5000 == 0:
                print('saving checkpoint')
                saver.save(sess, ckecpoint_file + '/model.ckpt', g_step_)

        total_timer.toc()
        sys.stdout.write('\n')
        print('>> mean loss', t_loss)
        print('curent speed: ', total_timer.average_time, 'remain time: ', total_timer.remain(g_step_, epoch_step*max_epoch))

        if g_step_%5000 == 0:
            print('saving checkpoint')
            saver.save(sess, ckecpoint_file + '/model.ckpt', g_step_)
