import numpy as np
import xml.etree.ElementTree as ET
import os
import cv2
import pickle
import copy
import retinanet_config as cfg
import random
from part.aug_python import _crop
from data_aug.data_aug import *
from data_aug.bbox_util import *
from part.gridmask import Grid

class pascal_voc(object):
    def __init__(self, phase, flipped, img_path, label_path, img_txt, is_training):
        self.is_training = is_training
        self.img_path = img_path
        self.label_path = label_path
        self.img_txt = img_txt
        self.cache_path = cfg.cache_path
        self.img_size = cfg.image_size
        self.batch_size = cfg.batch_size
        self.classes = cfg.classes
        self.class_to_ind = dict(zip(self.classes, range(len(self.classes))))
        self.flipped = flipped
        self.gt_labels = None
        self.phase = phase
        self.epoch = 1
        self.corsor = 0
        self.data_augmentation()
        
    def load_annotation(self,index):
        annotation_dir = os.path.join(self.label_path,index+'.xml')
        img_dir = os.path.join(self.img_path,index+'.jpg')
        img = cv2.imread(img_dir)
        print(img_dir)

        y, x = img.shape[0:2]
        resize_scale_x = self.img_size/x
        resize_scale_y = self.img_size/y
        tree = ET.parse(annotation_dir)
        objs = tree.findall('object')
        boxes_lenth = len(objs)
        label = np.zeros((boxes_lenth,5))
        
        i = 0
        for obj in objs:
            bbox = obj.find('bndbox')
            x1 = (float(bbox.find('xmin').text)) * resize_scale_x
            y1 = (float(bbox.find('ymin').text)) * resize_scale_y
            x2 = (float(bbox.find('xmax').text)) * resize_scale_x
            y2 = (float(bbox.find('ymax').text)) * resize_scale_y
            ind_clas = self.class_to_ind[obj.find('name').text.lower().strip()]
            boxes = [int(x1), int(y1), int(x2), int(y2)]
            label[i,0] = ind_clas
            label[i,1:5] = boxes
            i = i + 1
        return label,len(label)

        
    def read_image(self,imgnm, bboxes, flipped = False):
        img = cv2.imread(imgnm)
        img = cv2.resize(img,(self.img_size,self.img_size))
#         print('  :', bboxes.shape)
        ######################grid mask######################
        if cfg.gridmask and self.is_training:
            p = random.random()
            if p>0.7:
                grid_mask = Grid(use_h=True, use_w=True, use_object_drop = True)
                img, _, _ = grid_mask.__call__(img, bboxes)
        
        ######################grid mask######################
        #flip first
        if flipped == True:
            img = img[:, ::-1, :]
        #then bgrtorgb
        img=img[:,:,::-1]
        img=img.astype(np.float32, copy=False)
        
        #######################random crop#####################
        if cfg.random_crop and self.is_training:
            box_label = bboxes[:,:4]
            class_label = bboxes[:, 4]

            image_t, boxes_t, labels_t = _crop(img, box_label, class_label)
            t_h, t_w = image_t.shape[:2]
            t_x1 = boxes_t[:, 0]*self.img_size/t_w
            t_y1 = boxes_t[:, 1]*self.img_size/t_h
            t_x2 = boxes_t[:, 2]*self.img_size/t_w
            t_y2 = boxes_t[:, 3]*self.img_size/t_h
            boxes_t = np.vstack((t_x1, t_y1, t_x2, t_y2)).transpose()
            bboxes = np.append(boxes_t, labels_t.reshape(-1, 1), axis = 1)

            img = image_t
            img = cv2.resize(img,(self.img_size,self.img_size))
        #######################random crop#####################  
        
        #######################normallize#####################  
#         PIXEL_MEANS =(0.485, 0.456, 0.406)  #RGB format mean and variances
#         PIXEL_STDS = (0.229, 0.224, 0.225)
#         img/=255.0
#         img-=np.array(PIXEL_MEANS)
#         img/=np.array(PIXEL_STDS)
        #######################normallize##################### 
    
        ######################augment stratage 2###############
        if cfg.other_aug and self.is_training:
            p = random.random()
            if p>0.5:
                seq = Sequence([Rotate(90), RandomHorizontalFlip(), RandomScale(), RandomTranslate(), RandomShear()])
                img_, bboxes_ = seq(img, bboxes)
                if bboxes_.shape[0] != 0:
                    bboxes = bboxes_
                    img = img_
        ######################augment stratage 2###############    

        
        mean = np.array([123.68, 116.779, 103.979])
        mean = mean.reshape(1,1,3)
        img = img - mean
        return img, bboxes
    
    def load_labels(self):
        gt_labels = []
        d = 0
        cache_file = os.path.join(
            self.cache_path, 'pascal_' + self.phase + '_gt_labels.pkl')
        if os.path.isfile(cache_file):
            print('Loading gt_labels from: ' + cache_file)
            with open(cache_file, 'rb') as f:
                gt_labels = pickle.load(f)
            return gt_labels
        
        if not os.path.exists(self.cache_path):
            os.makedirs(self.cache_path)
        with open(self.img_txt, 'r') as f:
            self.image_index = [x.strip() for x in f.readlines()]
        for filename in self.image_index:
            d = d + 1
            print(d)
            ind = filename.split('.')[0]
            label,num = self.load_annotation(ind)
            imgnm = os.path.join(self.img_path, filename + '.jpg')
            print(imgnm)
            gt_labels.append({
                     'label' : label,
                     'img_dir' : imgnm,
                     'flipped' : False
                    })
        self.index = d
        print('Saving gt_labels to: ' + cache_file)
        with open(cache_file, 'wb') as f:
            pickle.dump(gt_labels, f)
        return gt_labels
            
    def data_augmentation(self):
        gt_labels = self.load_labels()
        if self.flipped:
            print('Appending flipped example...')
            gt_labels_cp = copy.deepcopy(gt_labels)
            for ind in range(len(gt_labels_cp)):
                gt_labels_cp[ind]['flipped'] = True
                label_len = len(gt_labels_cp[ind]['label'])
                for i in range(label_len):
                    change1_ = self.img_size - \
                        gt_labels_cp[ind]['label'][i,1]
                    change3_ = self.img_size - \
                        gt_labels_cp[ind]['label'][i,3]
                    gt_labels_cp[ind]['label'][i,1] = \
                        change3_
                    gt_labels_cp[ind]['label'][i,3] = \
                        change1_
            
            gt_labels += gt_labels_cp
        np.random.shuffle(gt_labels)
        self.gt_labels = gt_labels
        return gt_labels
    

        
        
    def get(self,):

        batch_imnm = []
        num_boxes = []
        images = []
        labels = np.zeros((self.batch_size,80,5))
        count = 0
        while count < self.batch_size:
            imnm = self.gt_labels[self.corsor]['img_dir']
            flipped = self.gt_labels[self.corsor]['flipped']

            label = self.gt_labels[self.corsor]['label']

            label = np.append(label[:,1:],label[:,0].reshape(-1, 1), axis = 1)

            image, label = self.read_image(imnm, label, flipped)
            label = np.append(label[:,4].reshape(-1, 1), label[:,:4], axis = 1)
            
            labels[count,:label.shape[0],:] = label
            num_boxes.append(label.shape[0])
            images.append(image)

            batch_imnm.append(imnm)
            count += 1
            self.corsor += 1
            if self.corsor >= len(self.gt_labels):
                np.random.shuffle(self.gt_labels)
                self.corsor = 0
                self.epoch += 1
                
      
        return np.asarray(images), labels, batch_imnm, num_boxes

#use example:        
p = pascal_voc(phase='train', flipped=True, img_path = cfg.train_img_path, label_path=cfg.train_label_path, img_txt=cfg.train_img_txt, is_training=True)
_ = p.load_labels()