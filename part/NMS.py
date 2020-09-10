import numpy as np
import tensorflow as tf

def gpu_nms(boxes, scores, num_classes, max_boxes=50, score_thresh=0.5, nms_thresh=0.5):
    """
    Perform NMS on GPU using TensorFlow.

    params:
        boxes: tensor of shape [1, 10647, 4] # 10647=(13*13+26*26+52*52)*3, for input 416*416 image
        scores: tensor of shape [1, 10647, num_classes], score=conf*prob
        num_classes: total number of classes
        max_boxes: integer, maximum number of predicted boxes you'd like, default is 50
        score_thresh: if [ highest class probability score < score_threshold]
                        then get rid of the corresponding box
        nms_thresh: real value, "intersection over union" threshold used for NMS filtering
    """

    boxes_list, label_list, score_list = [], [], []
    max_boxes = tf.constant(max_boxes, dtype='int32')

    # since we do nms for single image, then reshape it
    boxes = tf.reshape(boxes, [-1, 4]) # '-1' means we don't konw the exact number of boxes
    score = tf.reshape(scores, [-1, num_classes])

    # Step 1: Create a filtering mask based on "box_class_scores" by using "threshold".
    mask = tf.greater_equal(score, tf.constant(score_thresh))
    # Step 2: Do non_max_suppression for each class
    for i in range(num_classes):
        # Step 3: Apply the mask to scores, boxes and pick them out
        filter_boxes = tf.boolean_mask(boxes, mask[:,i])
        filter_score = tf.boolean_mask(score[:,i], mask[:,i])
        nms_indices = tf.image.non_max_suppression(boxes=filter_boxes,
                                                   scores=filter_score,
                                                   max_output_size=max_boxes,
                                                   iou_threshold=nms_thresh, name='nms_indices')
        label_list.append(tf.ones_like(tf.gather(filter_score, nms_indices), 'int32')*i)
        boxes_list.append(tf.gather(filter_boxes, nms_indices))
        score_list.append(tf.gather(filter_score, nms_indices))

    boxes = tf.concat(boxes_list, axis=0)
    score = tf.concat(score_list, axis=0)
    label = tf.concat(label_list, axis=0)

    return boxes, score, label



def NMS(pred_box, pred_score, iou, max_keep):
    '''
    for a single input and a certain class, the predict boxes is a tensor with shape of
    [M, 4] and the predict scores is a tensor with shape of [M,]'''
    
    x1 = pred_box[:,0]
    y1 = pred_box[:,1]
    x2 = pred_box[:,2]
    y2 = pred_box[:,3]
    
    w = np.maximum(0, x2-x1 + 1)
    h = np.maximum(0, y2-y1 + 1)
    area = w*h
    
    inds = pred_score.argsort()[::-1]
    
    keep = []
    while inds.size > 0:
        i = inds[0]
        keep.append(i)
        
        xx1 = np.maximum(x1[i], x1[inds[1:]])
        yy1 = np.maximum(y1[i], y1[inds[1:]])
        xx2 = np.maximum(x2[i], x2[inds[1:]])
        yy2 = np.maximum(y2[i], y2[inds[1:]])
        interw = np.maximum(0, xx2-xx1)
        interh = np.maximum(0, yy2-yy1)
        inter = interw*interh
        iou1 = inter/(area[i] + area[inds[1:]] + inter)
        inds1 = np.where(iou1<iou)[0]
        inds = inds[inds1+1]
    return keep[:max_keep]


def cpu_nms(pred_box, pred_score, score_th, iou_th, max_keep):
    '''
    for pred raw result implementing NMS
    pred_box:[M,4]
    pred_score:[M,classes]
    '''
    nms_box, nms_score, nms_label = [], [], []
    for i in range(pred_score.shape[-1]):
        pred_score_i_mask = np.where(pred_score[:,i]>score_th)[0]
        pred_score_i = pred_score[:,i][pred_score_i_mask]
        pred_box_i = pred_box[pred_score_i_mask]
        nms_inds = NMS(pred_box_i, pred_score_i, iou_th, max_keep)
        nms_box.append(pred_box_i[nms_inds])
        nms_score.append(pred_score_i[nms_inds])
        nms_label.append(np.ones(len(nms_inds), dtype='int32')*i)
    nms_box = np.concatenate(nms_box, axis = 0)
    nms_score = np.concatenate(nms_score, axis = 0)
    nms_label = np.concatenate(nms_label, axis = 0)
    return nms_box, nms_score, nms_label
        
    
