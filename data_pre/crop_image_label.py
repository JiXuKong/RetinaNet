import numpy as np
import xml.etree.ElementTree as ET
import xml.dom.minidom
import cv2
import os

def iou1_(box1, box2):
    x_min = np.maximum(box1[:,0], box2[:,0])
    y_min = np.maximum(box1[:,1], box2[:,1])
    x_max = np.minimum(box1[:,2], box2[:,2])
    y_max = np.minimum(box1[:,3], box2[:,3])
    return x_min, y_min, x_max, y_max


def iou_(box1, box2):
    '''box1:gt, array, N, 4,
       box2:crop box, array, 1, 4
       '''
#     print(box1.shape, box2.shape)
    x_min = np.maximum(box1[:,0], box2[:,0])
    y_min = np.maximum(box1[:,1], box2[:,1])
    x_max = np.minimum(box1[:,2], box2[:,2])
    y_max = np.minimum(box1[:,3], box2[:,3])
    h = np.maximum((y_max - y_min), 0)
    w = np.maximum((x_max - x_min), 0)
    area = w*h
    area1 = (box1[:, 2] - box1[:, 0])*(box1[:, 3] - box1[:, 1])
    area2 = (box2[:, 2] - box2[:, 0])*(box2[:, 3] - box2[:, 1])
#     print(area1, area2, area)
    iou = area/(area1 + area2 - area)
    return iou


def read_xml(path):
    tree = ET.parse(path)
    objs = tree.findall('object')
    box = []
    labels = []
    for obj in objs:
        bbox = obj.find('bndbox')
        label = obj.find('name').text.lower().strip()
        x1 = float(bbox.find('xmin').text)
        y1 = float(bbox.find('ymin').text)
        x2 = float(bbox.find('xmax').text)
        y2 = float(bbox.find('ymax').text)
        box.append([x1, y1, x2, y2])
        labels.append(label)
    return box, labels


def txt_2_xml(imageList, save_path, name):
    ss = imageList['object']
    for i in range(len(ss)):
        img_w = imageList['img_w']
        img_h = imageList['img_h']
        img_path = imageList['image_path']
        img_name = imageList['img_name']
        final_box = imageList['object']
    
        #计算boxe个数
        obj_len = len(final_box)
        #在内存中创建一个空的文档
        doc = xml.dom.minidom.Document() 
        #创建一个根节点Managers对象
        root_annotation = doc.createElement('annotation')
        doc.appendChild(root_annotation)
        root_folder = doc.createElement('folder')
        root_folder.appendChild(doc.createTextNode('images'))

        root_filename = doc.createElement('filename')
        root_filename.appendChild(doc.createTextNode(img_name))

        root_path = doc.createElement('path')
        root_path.appendChild(doc.createTextNode(img_path))

        root_annotation.appendChild(root_folder)
        root_annotation.appendChild(root_filename)
        root_annotation.appendChild(root_path)

        source = doc.createElement('source')
        nodesource = doc.createElement('database')
        nodesource.appendChild(doc.createTextNode('Unknown'))
        source.appendChild(nodesource)

        size = doc.createElement('size')
        nodesize_width = doc.createElement('width')
        nodesize_width.appendChild(doc.createTextNode(str(img_w)))
        nodesize_height = doc.createElement('height')
        nodesize_height.appendChild(doc.createTextNode(str(img_h)))
        nodesize_depth = doc.createElement('depth')
        nodesize_depth.appendChild(doc.createTextNode('3'))
        size.appendChild(nodesize_width)
        size.appendChild(nodesize_height)
        size.appendChild(nodesize_depth)

        root_annotation.appendChild(source)
        root_annotation.appendChild(size)
    
        for i in range(obj_len):

            object1 = doc.createElement('object')
            nodeobject_name = doc.createElement('name')
            nodeobject_name.appendChild(doc.createTextNode(final_box[i][0]))
            nodeobject_pose = doc.createElement('pose')
            nodeobject_pose.appendChild(doc.createTextNode('Unspecified'))
            nodeobject_truncated = doc.createElement('truncated')
            nodeobject_truncated.appendChild(doc.createTextNode('0'))
            nodeobject_difficult = doc.createElement('difficult')
            nodeobject_difficult.appendChild(doc.createTextNode('0'))

#             num_str = imageList['object'][i][0]
#             num_axis = num_str.split()
            xmin = str(final_box[i][1])
            xmax = str(final_box[i][3])
            ymin = str(final_box[i][2])
            ymax = str(final_box[i][4])

            nodeobject_bndbox = doc.createElement('bndbox')
            nodeobject_bndbox_xmin = doc.createElement('xmin')
            nodeobject_bndbox_xmin.appendChild(doc.createTextNode(xmin))
            nodeobject_bndbox_ymin = doc.createElement('ymin')
            nodeobject_bndbox_ymin.appendChild(doc.createTextNode(ymin))
            nodeobject_bndbox_xmax = doc.createElement('xmax')
            nodeobject_bndbox_xmax.appendChild(doc.createTextNode(xmax))
            nodeobject_bndbox_ymax = doc.createElement('ymax')
            nodeobject_bndbox_ymax.appendChild(doc.createTextNode(ymax))
            nodeobject_bndbox.appendChild(nodeobject_bndbox_xmin)
            nodeobject_bndbox.appendChild(nodeobject_bndbox_ymin)
            nodeobject_bndbox.appendChild(nodeobject_bndbox_xmax)
            nodeobject_bndbox.appendChild(nodeobject_bndbox_ymax)
            object1.appendChild(nodeobject_name)
            object1.appendChild(nodeobject_pose)
            object1.appendChild(nodeobject_truncated)
            object1.appendChild(nodeobject_difficult)
            object1.appendChild(nodeobject_bndbox)
            root_annotation.appendChild(object1)
        #object1.appendChild(nodeobject_bndbox)

            fp = open(os.path.join(save_path, name + '.xml'), 'w')
#             fp = open('F:\\linshi\\xml\\split_tamian\\test\\' + img_name.split('.')[0]+ '.xml', 'w')
        doc.writexml(fp, indent='\t', addindent='\t', newl='\n', encoding="utf-8")
    
def crop_(path, n, cover_pix, cropimg_path, cropxml_path):
    k = 0
    for fil in os.listdir(path):
        if fil.split('.')[1] == 'jpg':
            jpg_path = os.path.join(path, fil)
            xml_path = os.path.join(path, fil.split('.')[0] + '.xml')
            img = cv2.imread(jpg_path)
#             h = y, w = x
            y, x = img.shape[:2]
            box, labels = read_xml(xml_path)
#             print(box)
            for i in range(n):
                for j in range(n):
                    crop_image = img[i*(y//n-cover_pix):(i+1)*(y//n), j*(x//n-cover_pix):(j+1)*(x//n)]
#                     cv2.imwrite(os.path.join(cropimg_path, str(k) + '.jpg'), crop_image)
                    x1, y1, x2, y2 = j*(x//n-cover_pix), i*(y//n-cover_pix), (j+1)*(x//n), (i+1)*(y//n)
                    box1 = np.array(box)
#                     print(x1, y1, x2, y2)
                    box2 = np.array([[x1, y1, x2, y2]])
                    iou = iou_(box1, box2)
#                     print(iou)
                    box_ind = np.where(iou > 0)
                    crop_box = box1[box_ind]
                    crop_labels = labels[box_ind]
#                     print('crop_box', crop_box)
                    x_min, y_min, x_max, y_max = iou1_(crop_box, box2)
#                     print(x_min, y_min, x_max, y_max)
                    x_min_ = x_min - x1
                    y_min_ = y_min - y1
                    x_max_ = x_max - x1
                    y_max_ = y_max - y1
                    #截后的框，x,y在新的图片中从0算起
                    box3 = np.hstack((x_min_.reshape(-1, 1), y_min_.reshape(-1, 1), x_max_.reshape(-1, 1), y_max_.reshape(-1, 1)))
                    box4 = np.hstack((x_min.reshape(-1, 1), y_min.reshape(-1, 1), x_max.reshape(-1, 1), y_max.reshape(-1, 1)))   
                    #截后的边缘框和没截的边缘框的iou
                    iou1 = iou_(crop_box, box4)
                    crop_index = np.where(iou1>0.5)
                    valid_box = box3[crop_index]
                    valid_labels = crop_labels[crop_index]
#                     print(valid_box)
                    saved_box = []
                    for ii in range(valid_box.shape[0]):
                        bbox = [valid_labels[ii], str(valid_box[ii][0]), str(valid_box[ii][1]), str(valid_box[ii][2]), str(valid_box[ii][3])]
                        saved_box.append(bbox)
                    imageList = {
                        'pose':'Unspecified',
                        'truncated':0,
                        'difficult':0,
                        'img_w':x2-x1,
                        'img_h':y2-y1,
                        'image_path':jpg_path,
                        'img_name':str(k),
                        'object':saved_box
                                }
                    txt_2_xml(imageList, cropxml_path, str(k))
                    if len(saved_box) >0:
                        cv2.imwrite(os.path.join(cropimg_path, str(k) + '.jpg'), crop_image)
                    k = k + 1
