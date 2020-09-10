import cv2
import numpy as np
import random
import math

def matrix_iou(a,b):

    """
    return iou of a and b, numpy version for data augenmentation
    
    """
    lt = np.maximum(a[:, np.newaxis, :2], b[:, :2])
    rb = np.minimum(a[:, np.newaxis, 2:], b[:, 2:])

    area_i = np.prod(rb - lt, axis=2) * (lt < rb).all(axis=2)
    area_a = np.prod(a[:, 2:] - a[:, :2], axis=1)
    area_b = np.prod(b[:, 2:] - b[:, :2], axis=1)
    return area_i / (area_a[:, np.newaxis] + area_b - area_i)



def _crop(image, boxes, labels):
    height, width, = image.shape[:2]    
    if len(boxes.shape) == 0:
        return image, boxes, labels
    
    while True:
        mode = random.choice((
            None,
            (0.1, None),
            (0.3, None),
            (0.5, None),
            (0.7, None),
            (0.9, None),
            (None, None),
        ))
        if mode is None:
            return image, boxes, labels
        
        
        min_iou, max_iou = mode
        if min_iou is None:
            min_iou = float('-inf')
        if max_iou is None:
            max_iou = float('inf')
            
        for _ in range(50):
            scale = random.uniform(0.3, 1.)
            min_ratio = max(0.5, scale*scale)
            max_ratio = min(2, 1./scale/scale)
            ratio = math.sqrt(random.uniform(min_ratio, max_ratio))
            w = int(scale*ratio*width)
            h = int((scale/ratio)*height)
            
            l = random.randrange(width - w)
            t = random.randrange(height - h)
            
            roi = np.array((l, t, l + w, t + h))
            iou = matrix_iou(boxes, roi[np.newaxis])
            
            if not (min_iou <= iou.min() and iou.max() <= max_iou):
                continue

            image_t = image[roi[1]:roi[3], roi[0]:roi[2]]

            centers = (boxes[:, :2] + boxes[:, 2:]) / 2
            mask = np.logical_and(roi[:2] < centers, centers < roi[2:]) \
                .all(axis=1)
            boxes_t = boxes[mask].copy()
            labels_t = labels[mask].copy()
            if len(boxes_t) == 0:
                continue


            boxes_t[:, :2] = np.maximum(boxes_t[:, :2], roi[:2])
            boxes_t[:, :2] -= roi[:2]
            boxes_t[:, 2:] = np.minimum(boxes_t[:, 2:], roi[2:])
            boxes_t[:, 2:] -= roi[:2]
            return image_t, boxes_t, labels_t

    
