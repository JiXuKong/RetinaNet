import numpy as np
# import tensorflow as tf
import retinanet_config as cfg

scale = np.array([2**0, 2**(1/3), 2**(2/3)])
aspect_ratio = np.array([1/2, 1, 2])

def anchor_(base_size, scale, aspect_ratio):
    area = base_size*base_size*scale**2
    w = np.sqrt(area.reshape((area.shape[0],1))/aspect_ratio.reshape(1,aspect_ratio.shape[0]))
    h = aspect_ratio*w
    w = w.transpose()
    h = h.transpose()
    w = w.reshape(-1)
    h = h.reshape(-1)
    anchor = np.vstack((-w/2, -h/2, w/2, h/2)).transpose()
    return anchor

def generate_anchor_(base_size, scale, aspect_ration, feature_size):
    base_anchor = anchor_(base_size, scale, aspect_ration)
    stride = [cfg.image_size//feature_size[0], cfg.image_size//feature_size[1]]
    grid = np.array([[j*stride[0]+stride[0]/2, i*stride[1]+stride[1]/2, j*stride[0]+stride[0]/2, i*stride[1]+stride[1]/2] for i in range(feature_size[1]) for j in range(feature_size[0])])
    generate_anchor = grid.reshape((-1, 1 ,4)) + base_anchor.reshape(1, -1, 4)
    generate_anchor = generate_anchor.reshape(-1, 4)
#     normal to (0,1)
    
    return generate_anchor