import numpy as np
import tensorflow as tf
import retinanet_config as cfg
def Soomth_L1_loss(pred_target, label_target, inner_weight, outside_weight):
    inner_weight = tf.reshape(inner_weight, [-1, 4])
    outside_weight = tf.reshape(outside_weight, [-1, 4])
    pred_target = tf.reshape(pred_target, [-1, 4])
    label_target = tf.reshape(label_target, [-1, 4])
    inner_ = (pred_target - label_target)*inner_weight
    in_box_diff = tf.abs(inner_ )
    judge_mask = tf.stop_gradient(tf.to_float(tf.less(in_box_diff, 1)))
    in_loss_box = tf.pow(inner_, 2) * (1 / 2.) * judge_mask \
                      + (in_box_diff - (0.5 )) * (1. - judge_mask)
    outside_loss = tf.reduce_mean(tf.reduce_sum(outside_weight*in_loss_box, axis = 1))
    return outside_loss*1e2

def softmax_crossentropy_loss(pred_class, label, classes_num):
    label = tf.reshape(label, [-1, 1])
    one_hot_label = tf.one_hot(label, classes_num)
    pred_class = tf.reshape(label, [-1, classes_num])
    pred_sotmax = tf.nn.softmax(pred_class)
    loss = tf.reduce.mean(tf.reduce_sum(-one_hot_label*tf.log(pred_sotmax), axis = 1), axis = 0)
    return loss

def focal_loss(pred_class, label, classes_num, a, b):
    '''
    pred_class:[batch, anchornum]
    label:[batch, anchornum]
    a: class weight
    b:hard example weight
    '''
    label = tf.reshape(label, [-1,])
    one_hot_label = tf.one_hot(label, classes_num)
    s1 = tf.ones_like(label, dtype = tf.float32)*tf.constant(a)
    s2 = tf.ones_like(label, dtype = tf.float32)*(1-tf.constant(a))
    l_s = tf.where(label>0, s1, s2)
    p_num = tf.reduce_sum(tf.to_float(tf.less(0, label)))
    weight_label = tf.multiply(tf.reshape(l_s,[-1,1]), tf.cast(one_hot_label, dtype = tf.float32))
    pred_class = tf.reshape(pred_class, [-1, classes_num])
    pred_sigmoid = tf.nn.sigmoid(pred_class)
    focal_loss = tf.reduce_sum(tf.reduce_sum(-weight_label*tf.pow((1-pred_sigmoid), tf.constant(b, dtype = tf.float32))*tf.log(pred_sigmoid), axis = 1), axis = 0)/(p_num*100)
    return focal_loss


def Soomth_L1_loss_pru_tf(pred_target, label_target, bestmatch_an_inds):
    pred_target = tf.reshape(pred_target, [-1, 4])
    pred_target = tf.gather(pred_target, bestmatch_an_inds)
    target = pred_target - label_target
    target = tf.abs(target)
    judge_mask = tf.stop_gradient(tf.to_float(tf.less(target, 1)))
    loss_box = tf.pow(target, 2) * (1 / 2.) * judge_mask \
                      + (target - (0.5 )) * (1. - judge_mask)
    outside_loss = tf.reduce_mean(tf.reduce_sum(loss_box, axis = 1))
    return outside_loss


# def focal_loss_pru_tf(pred_class, bestmatch_an_inds, bestmatch_gt_label, neg_an_inds, classes_num, a, b):
#     a = tf.constant(a, dtype = tf.float32)
#     gama = tf.constant(b, dtype = tf.float32)
#     pred_class = tf.reshape(pred_class, [-1, classes_num])
# #     pred_sigmoid = pred_class
#     pred_pos_class = tf.gather(pred_class, bestmatch_an_inds)
#     pred_neg_class = tf.gather(pred_class, neg_an_inds)
#     pos_label = tf.reshape(bestmatch_gt_label, [-1,])
#     pos_label = tf.cast(pos_label, dtype = tf.int32)
#     neg_label = tf.reshape(neg_an_inds, [-1,])
#     neg_label = tf.zeros_like(neg_label, dtype = tf.int32)
#     pos_label = tf.one_hot(pos_label, classes_num)
#     neg_label = tf.one_hot(neg_label, classes_num)    
#     pos_ce = tf.nn.softmax_cross_entropy_with_logits(labels=pos_label, logits=pred_pos_class)
#     neg_ce = tf.nn.softmax_cross_entropy_with_logits(labels=neg_label, logits=pred_neg_class)
#     if cfg.focal_loss:
#         pos_softmax = tf.nn.softmax(pred_pos_class)
#         neg_softmax = tf.nn.softmax(pred_neg_class)
#         pos_softmax = tf.clip_by_value(pos_softmax, 1e-8, 1.)
#         neg_softmax = tf.clip_by_value(neg_softmax, 1e-8, 1.)
#         p_num = tf.cast(tf.shape(bestmatch_gt_label)[0], dtype = tf.float32)
#         pos_loss = a*tf.reduce_sum(pos_ce*tf.reduce_sum(pos_label*tf.pow(1-pos_softmax, gama), axis = -1)/4.0, axis = 0)/p_num
#         neg_loss = (1-a)*tf.reduce_sum(neg_ce*tf.reduce_sum(neg_label*tf.pow(1-neg_softmax, gama), axis = -1)/4.0, axis = 0)/p_num
#         total_focal = pos_loss+neg_loss
#     else:
#         pos_loss = tf.reduce_mean(pos_ce)
#         neg_loss = tf.reduce_mean(neg_ce)
#         total_focal = pos_loss + neg_loss
#         p_num = 256
#     return total_focal, pos_loss, neg_loss, p_num, pos_ce, neg_ce
    

def focal_loss_pru_tf(pred_class, bestmatch_an_inds, bestmatch_gt_label, neg_an_inds, classes_num, a, b):
    a = tf.constant(a, dtype = tf.float32)
    pred_class = tf.reshape(pred_class, [-1, classes_num])
#     pred_sigmoid = pred_class
#     pred_sigmoid = tf.nn.sigmoid(pred_class)
    pred_pos_class = tf.gather(pred_class, bestmatch_an_inds)
    pred_neg_class = tf.gather(pred_class, neg_an_inds)
    pos_label = tf.reshape(bestmatch_gt_label, [-1,])
    pos_label = tf.cast(pos_label, dtype = tf.int32)
    neg_label = tf.reshape(neg_an_inds, [-1,])
    neg_label = tf.zeros_like(neg_label, dtype = tf.int32)
    
    pos_label = tf.one_hot(pos_label, classes_num)
    neg_label = tf.one_hot(neg_label, classes_num)  
    
    
    
    neg_ce = tf.nn.sigmoid_cross_entropy_with_logits(labels=neg_label, logits=pred_neg_class)
    pos_ce = tf.nn.sigmoid_cross_entropy_with_logits(labels=pos_label, logits=pred_pos_class)
    
    neg_sigmoid = tf.clip_by_value(tf.nn.sigmoid(pred_neg_class), 1e-8, 1.0)
    pos_sigmoid = tf.clip_by_value(tf.nn.sigmoid(pred_pos_class), 1e-8, 1.0)
    
    pos_outer_pred = 0.0#pos_label*pos_sigmoid + (1-pos_label)*(1-pos_sigmoid)
    neg_outer_pred = 0.0#neg_label*neg_sigmoid + (1-neg_label)*(1-neg_sigmoid)
    
    pos_alpha_weight = 0.0#pos_label*a + (1-pos_label)*(1-a)
    neg_alpha_weight = neg_label*a + (1-neg_label)*(1-a)
    
    gama = tf.constant(b, dtype = tf.float32)
    p_num = tf.cast(tf.shape(bestmatch_gt_label)[0], dtype = tf.float32)
    
    pos_loss = 0.0#tf.reduce_sum(tf.reduce_sum(pos_alpha_weight*tf.pow(1-pos_outer_pred, gama)*pos_ce, axis = -1), axis = 0)/p_num
    neg_loss = 0.0#tf.reduce_sum(tf.reduce_sum(neg_alpha_weight*tf.pow(1-neg_outer_pred, gama)*neg_ce, axis = -1), axis = 0)/p_num
    
    focal_loss = tf.reduce_mean(tf.reduce_sum(pos_sigmoid, axis = -1),axis=[0])+tf.reduce_mean(tf.reduce_sum(neg_sigmoid, axis = -1),axis=[0])#pos_loss+neg_loss
    return 10*focal_loss, pos_loss, neg_loss, p_num, tf.reduce_sum(pos_sigmoid, axis = -1), tf.reduce_sum(neg_sigmoid, axis = -1)
    

# def focal_loss_pru_tf(pred_class, bestmatch_an_inds, bestmatch_gt_label, neg_an_inds, classes_num, a, b):
#     a = tf.constant(a, dtype = tf.float32)
#     pred_class = tf.reshape(pred_class, [-1, classes_num])
# #     pred_sigmoid = pred_class
# #     pred_sigmoid = tf.nn.sigmoid(pred_class)
#     pred_pos_class = tf.gather(pred_class, bestmatch_an_inds)
#     pred_neg_class = tf.gather(pred_class, neg_an_inds)
#     pos_label = tf.reshape(bestmatch_gt_label, [-1,])
#     pos_label = tf.cast(pos_label, dtype = tf.int32)
#     neg_label = tf.reshape(neg_an_inds, [-1,])
#     neg_label = tf.zeros_like(neg_label, dtype = tf.int32)
#     concat_label = tf.concat([neg_label, pos_label], axis = 0)
#     concat_pred = tf.concat([pred_neg_class, pred_pos_class], axis = 0)
#     one_hot_label = tf.one_hot(concat_label, classes_num)
#     ce = tf.nn.sigmoid_cross_entropy_with_logits(labels=one_hot_label, logits=concat_pred)
#     concat_pred = tf.nn.sigmoid(concat_pred)
#     concat_pred = tf.clip_by_value(concat_pred, 1e-8, 1.)
# #     ce = -one_hot_label*tf.log(concat_pred)
#     outer_pred = one_hot_label*concat_pred + (1-one_hot_label)*(1-concat_pred)
#     alpha_weight = one_hot_label*a + (1-one_hot_label)*(1-a)
#     gama = tf.constant(b, dtype = tf.float32)
#     p_num = tf.cast(tf.shape(bestmatch_gt_label)[0], dtype = tf.float32)
#     focal_loss = tf.reduce_sum(tf.reduce_sum(alpha_weight*tf.pow(1-outer_pred, gama)*ce, axis = -1), axis = 0)/p_num
#     return focal_loss
    
    
    