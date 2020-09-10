import tensorflow as tf
import numpy as np
import math
# from net.new_resnet import resnet, activa_conv_gn, conv_2d_same
from net.resnet_ import resnet_base
# from net.resnet_v1 import resnet_v1_50
from part.select_anchor import select_anchor, select_anchor_tf, select_anchor_tf_loss
from part.anchor import generate_anchor_
from part.losses import localization_loss, focal_loss
from part.regress_target import regress_target
from net.normalization import gn_, bn_
import retinanet_config as cfg
from part.training_target_creation import get_training_targets

slim = tf.contrib.slim
class Retinanet(object):
    def __init__(self, inpu_image, inpu_label, input_num_boxes):
        self.filter_list = cfg.filter_list
        self._bottleneck = cfg. _bottleneck
#         self.is_training = tf.placeholder(tf.bool, shape = [])
        self.is_training = True
        self.class_num = cfg.class_num
        self.anchors = cfg.anchors
        self.base_anchor = cfg.base_anchor
        self.scale = cfg.scale
        self.aspect_ratio = cfg.aspect_ratio
        self.feature_size = cfg.feature_size
        self.image_size = cfg.image_size
        self.alpha = cfg.alpha
        self.gama = cfg.gama
        self.class_weight = cfg.class_weight
        self.regress_weight = cfg.regress_weight
        self.l2_weight = cfg.l2_weight
        self.decay = cfg.decay
        self.pi = cfg.pi
        self.image = inpu_image#tf.placeholder(tf.float32, shape = [None, self.image_size, self.image_size, 3])
        self.label = inpu_label#tf.placeholder(tf.int32, shape = [None, None, 5])
        self.num_boxes = input_num_boxes
        self.anchor = self._generate_anchor()#tf.placeholder(tf.float32, shape = [None, 4])
        self.anchorlist = self._generate_anchor()
#         self.pred_classification_target_list, self.pred_regress_target_list = self.forward()
        self.total_loss, self.regular_loss, self.loc_loss, self.cls_loss, self.normalizer = self._loss()
    
    def upsample_layer(self, inputs, out_shape, scope):
        with tf.name_scope(scope):
            new_height, new_width = out_shape[0], out_shape[1]
            # NOTE: here height is the first
        #         inputs = tf.image.resize_nearest_neighbor(inputs, (new_height, new_width), align_corners=False, name=scope)

            channels = inputs.shape[3].value
            rate = 2
            x = tf.reshape(inputs, [cfg.batch_size//2, new_height//rate, 1, new_width//rate, 1, channels])
            x = tf.tile(x, [1, 1, rate, 1, rate, 1])
            x = tf.reshape(x, [cfg.batch_size//2, new_height, new_width, channels])
            return x
    
     def feature_extract(self):
        end_points = resnet_base(self.image, self.is_training, 'resnet_v1_50')
        return end_points
    def FPN_structure(self, projection_norm = True):
        end_points = self.feature_extract()
#         batch_norm_params = {
#           'is_training': self.is_training,
#           'decay': 0.9,
#           'epsilon': 1e-3,
#           'scale': batch_norm_scale,
#           'trainable': True,
#           'updates_collections': tf.GraphKeys.UPDATE_OPS,
#           'fused': None  # Use fused batch norm if possible.
#           }
        with tf.variable_scope('FeatureExtractor/resnet_v1_50/fpn'):
            with slim.arg_scope([slim.conv2d], weights_regularizer=slim.l2_regularizer(4e-3),
                            trainable=self.is_training, activation_fn=None):#, normalizer_fn=slim.batch_norm, normalizer_params=batch_norm_params):
                
                for level in range(5,2,-1):
                    end_points['p'+str(level)] = slim.conv2d(end_points['p'+str(level)], 256, [1, 1], trainable=self.is_training,
                                        weights_initializer=tf.truncated_normal_initializer(mean=0.0, stddev=0.03),
                                        scope='projection_'+str(level-2))

                #先做p5reducedim,在做p6，p7
                p6 = slim.conv2d(end_points['p'+str(5)], 256, [3, 3], trainable=self.is_training,
                                weights_initializer=tf.truncated_normal_initializer(mean=0.0, stddev=0.03),
                                biases_initializer=None,
                                padding="SAME",
                                stride=2,
                                scope='bottom_up_block5')
                p6 = bn_relu(p6, True, self.is_training, 'bottom_up_block5/BatchNorm')
                
                p7 = slim.conv2d(p6, 256, [3, 3], trainable=self.is_training,
                                weights_initializer=tf.truncated_normal_initializer(mean=0.0, stddev=0.03),
                                biases_initializer=None,
                                padding="SAME",
                                stride=2,
                                activation_fn=None,
                                scope='bottom_up_block6')
                p7 = bn_relu(p7, True, self.is_training, 'bottom_up_block6/BatchNorm')
                
                #p5, p4的上采样相加
                for level in range(5,3,-1):
                    plevel_up = self.upsample_layer(end_points['p'+str(level)], [end_points['p'+str(level-1)].get_shape().as_list()[1],
                                                                 end_points['p'+str(level-1)].get_shape().as_list()[2]],
                                                                 'p'+str(level)+'_upsample')
                    end_points['p'+str(level-1)] = tf.add(end_points['p'+str(level-1)], plevel_up, 'fuse_p'+str(level-1))


                for level in range(4,2,-1):
                    end_points['p'+str(level)] = slim.conv2d(end_points['p'+str(level)], 256, [3, 3], trainable=self.is_training,
                                    weights_initializer=tf.truncated_normal_initializer(mean=0.0, stddev=0.03),
                                    biases_initializer=None,
                                    padding="SAME",
                                    stride=1,
                                    scope='smoothing_'+str(level-2))

                
                
                p3 = end_points['p3']
                p4 = end_points['p4']
                p5 = end_points['p5']
                
            
            return [p3, p4, p5, p6, p7]
    

    #subnet权重共享
    def baseclassification_subnet(self, features, feature_level):
        reuse1 = tf.AUTO_REUSE
        for j in range(4):
            features = slim.conv2d(features, 256, [3,3], trainable=self.is_training,
                                   weights_initializer=tf.random_normal_initializer(mean=0.0, stddev=0.01),
                                   biases_initializer=None,
                                   stride=1,
                                   activation_fn=None,
                                   normalizer_fn= tf.identity,
                                   scope='ClassPredictionTower/conv2d_' + str(j),# + str(feature_level), 
                                  reuse=reuse1)
            features = bn_relu(features, True, self.is_training, 'ClassPredictionTower/conv2d_%d/BatchNorm/feature_%d' % (j, feature_level))

        class_feature_output = slim.conv2d(features, (self.class_num-1)*self.anchors, [3,3], trainable=self.is_training,
                                   weights_initializer=tf.random_normal_initializer(mean=0.0, stddev=0.01),
                                   biases_initializer=tf.constant_initializer(-math.log((1 - self.pi)/self.pi)),
                                   stride=1,
                                   activation_fn= None,
                                   scope='ClassPredictor', 
                                   reuse=reuse1)

        return class_feature_output
    def baseregression_subnet(self, features, feature_level):    
        reuse2 = tf.AUTO_REUSE
        for j in range(4):
            features = slim.conv2d(features, 256, [3,3], trainable=self.is_training,
                                   weights_initializer=tf.random_normal_initializer(mean=0.0, stddev=0.01),
                                   biases_initializer=None,
                                   stride=1,
                                   activation_fn=None,
                                   normalizer_fn= tf.identity,
                                   scope='BoxPredictionTower/conv2d_' + str(j),# + str(feature_level), 
                                   reuse=reuse2)
            features = bn_relu(features, True, self.is_training, 'BoxPredictionTower/conv2d_%d/BatchNorm/feature_%d' % (j, feature_level))

        regress_feature_output = slim.conv2d(features, 4*self.anchors, [3,3], trainable=self.is_training,
                                   weights_initializer=tf.random_normal_initializer(mean=0.0, stddev=0.01),
                                   stride=1,
                                   activation_fn=None,
                                   scope='BoxPredictor', 
                                   reuse=reuse2)

        return regress_feature_output
        
    def pred_subnet(self, fpn_features):
        cfeatures_ = []
        rfeatures_ = []
        with tf.variable_scope('WeightSharedConvolutionalBoxPredictor'):
            for i in range(3, len(fpn_features)+3):
                class_feature_output = self.baseclassification_subnet(fpn_features[i-3], i-3, )
                clas_shape = class_feature_output.get_shape().as_list()
                cfeatures_.append(tf.reshape(class_feature_output, [-1, clas_shape[1]*clas_shape[2]*self.anchors, (self.class_num-1)]))
                
                regress_feature_output = self.baseregression_subnet(fpn_features[i-3], i-3)
                reg_shape = regress_feature_output.get_shape().as_list()
                rfeatures_.append(tf.reshape(regress_feature_output, [-1, reg_shape[1]*reg_shape[2]*self.anchors, 4])) 

            return tf.concat(cfeatures_, axis = 1), tf.concat(rfeatures_, axis = 1)
    
    
    '''
    def classification_subnet(self, fpn_features):
        cfeatures_ = []
        with tf.variable_scope('class_subnet'):
            for i in range(len(fpn_features)):
                for j in range(3):
                    class_feature_output = fpn_features[i]
                    class_feature_output = activa_conv_gn(class_feature_output, 256, 3, 1,
                                                          1, 'level' + str(i) + 'conv' + str(j), True, self.is_training)
                class_feature_output = slim.conv2d(class_feature_output, self.class_num*self.anchors, [3, 3], trainable=self.is_training,
                                    weights_initializer=tf.contrib.keras.initializers.he_normal(),
                                    biases_initializer=tf.constant_initializer(-math.log((1 - self.pi) / self.pi)),
                                    padding='SAME',
                            activation_fn=tf.nn.sigmoid if not self.is_training else None,
                            scope='level' + str(i) + 'class_output')
                
                clas_shape = class_feature_output.get_shape().as_list()
                cfeatures_.append(tf.reshape(class_feature_output, [-1, clas_shape[1]*clas_shape[2], self.class_num*self.anchors]))
        return tf.concat(cfeatures_, axis = 1) 
        
    def regression_subnet(self, fpn_features):
        rfeatures_ = []
        with tf.variable_scope('regress_subnet'):
            for i in range(len(fpn_features)):
                for j in range(3):
                    regress_feature_output = fpn_features[i]
                    regress_feature_output = activa_conv_gn(regress_feature_output, 256, 3, 1,
                                                            1, 'level' + str(i) + 'conv' + str(j), True, self.is_training)
                regress_feature_output = slim.conv2d(regress_feature_output, 4*self.anchors, [3, 3], trainable=self.is_training,
                                    weights_initializer=tf.contrib.keras.initializers.he_normal(),
                                    padding='SAME', activation_fn=None, scope='level' + str(i) + 'regress_output')
                reg_shape = regress_feature_output.get_shape().as_list()
                rfeatures_.append(tf.reshape(regress_feature_output, [-1, reg_shape[1]*reg_shape[2], 4*self.anchors]))
        return tf.concat(rfeatures_, axis = 1)
    '''
    def _generate_anchor(self):
        anchorlist = []
        for i in range(len(self.base_anchor)):
            anchors = generate_anchor_(self.base_anchor[i], self.scale, self.aspect_ratio, self.feature_size[i])
            anchorlist.append(anchors)
#         self.anchorlist = anchorlist
        return np.concatenate(anchorlist, axis = 0)
#     def _generate_labels(self):
#         anchorlist = self._generate_anchor()
#         if cfg.use_select_anchor_py:
#             classgt, regressgt, inner_weight, outside_weight = tf.py_func(
#                     select_anchor,
#                     [self.label, anchorlist],
#                     [tf.int32, tf.float32, tf.float32, tf.float32])

#             return classgt, regressgt, inner_weight, outside_weight
#         else:
#             anchorlist = self.anchor
#             regress_label, bestmatch_an_inds, bestmatch_gt_label, neg_an_inds = select_anchor_tf(self.label, self.anchor)
#             return regress_label, bestmatch_an_inds, bestmatch_gt_label, neg_an_inds
    def forward(self):
        fpn_features = self.FPN_structure()
#         batch_norm_params = {
#           'is_training': self.is_training,
#           'decay': 0.9,
#           'epsilon': 1e-3,
#           'scale': batch_norm_scale,
#           'trainable': True,
#           'updates_collections': tf.GraphKeys.UPDATE_OPS,
#           'fused': None  # Use fused batch norm if possible.
#           }
#         with tf.variable_scope('Retinanet'):
#             fpn_features = self.FPN_structure()
        with slim.arg_scope([slim.conv2d], weights_regularizer=slim.l2_regularizer(4e-4), trainable=self.is_training, activation_fn=None):
            pred_classification_target_list, pred_regress_target_list = self.pred_subnet(fpn_features)
            return pred_classification_target_list, pred_regress_target_list

            
    def batch_target(self):
        def fn(x):
            boxes, labels, num_boxes = x
            boxes, labels = boxes[:num_boxes], labels[:num_boxes]

            reg_targets, cls_targets, matches = get_training_targets(
                self.anchor, boxes, labels,
                positives_threshold=0.5,
                negatives_threshold=0.4
            )
            return reg_targets, cls_targets, matches

        with tf.name_scope('target_creation'):
            reg_targets, cls_targets, matches = tf.map_fn(
                fn, [self.label[:, :, 1:], self.label[:, :, 0], self.num_boxes],
                dtype=(tf.float32, tf.int32, tf.int32),
                parallel_iterations=4,
                back_prop=False, swap_memory=False, infer_shape=True
            )
            return reg_targets, cls_targets, matches
        
    
    def _loss(self):
        self.pred_classification_target_list, self.pred_regress_target_list = self.forward()
        reg_targets, cls_targets, matches = self.batch_target()
        
        with tf.name_scope('losses'):
            # whether anchor is matched
            weights = tf.to_float(tf.greater_equal(matches, 0))
            with tf.name_scope('classification_loss'):
                class_predictions = tf.identity(self.pred_classification_target_list)
                # shape [batch_size, num_anchors, num_classes]
                cls_targets = tf.one_hot(cls_targets, self.class_num, axis=2)
                # shape [batch_size, num_anchors, num_classes + 1]
                # remove background
                cls_targets = tf.to_float(cls_targets[:, :, 1:])
                # now background represented by all zeros
                not_ignore = tf.to_float(tf.greater_equal(matches, -1))
                # if a value is `-2` then we ignore its anchor
                cls_losses = focal_loss(
                    class_predictions, cls_targets, weights=not_ignore,
                    gamma=cfg.gama, alpha=cfg.alpha)
                # it has shape [batch_size, num_anchors]
            with tf.name_scope('localization_loss'):
                encoded_boxes = tf.identity(self.pred_regress_target_list)
                # it has shape [batch_size, num_anchors, 4]
                loc_losses = localization_loss(encoded_boxes, reg_targets, weights)
                # shape [batch_size, num_anchors]
            with tf.name_scope('normalization'):
                matches_per_image = tf.reduce_sum(weights, axis=1)  # shape [batch_size]
                num_matches = tf.reduce_sum(matches_per_image, axis=0)  # shape []
                normalizer = tf.maximum(num_matches, 1.0)    
        
            loc_loss = tf.reduce_sum(loc_losses, axis=[0, 1])/normalizer
            cls_loss = tf.reduce_sum(cls_losses, axis=[0, 1])/normalizer
            #使用tf.loss
            tf.losses.add_loss(self.class_weight*cls_loss)
            tf.losses.add_loss(self.regress_weight*loc_loss)
            # add l2 regularization
            with tf.name_scope('weight_decay'):
                slim_losses = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
#                 trainable_variable = [var for var in tf.trainable_variables()]
# #                 for v in tf.trainable_variables():
# #                     print('l2: ', v.name)
#                 Kernels = [
#                     v for v in trainable_variable
#                     if v.name.split('/')[-1] == 'weights:0'
#                 ]
#                 for K in Kernels:
#                     print('l2: ', K.name)
#                     x = tf.multiply(self.l2_weight, tf.nn.l2_loss(K))
#                     tf.add_to_collection(tf.GraphKeys.REGULARIZATION_LOSSES, x)
                regularization_loss = tf.losses.get_regularization_loss()
            
            total_loss = tf.losses.get_total_loss(add_regularization_losses=True)
#             tf.summary.scalar('train/regularization_loss', regularization_loss)
#             tf.summary.scalar('train/localization_loss', regress_weight*loc_loss)
#             tf.summary.scalar('train/classification_loss', self.class_weight*cls_loss)
#             tf.summary.scalar('train/nummatches', normalizer/cfg.batch_size)
#             regular_loss = tf.add_n([tf.nn.l2_loss(var) for var in tf.trainable_variables()])
#             total_loss = self.class_weight*cls_loss + self.regress_weight*loc_loss + self.l2_weight*regular_loss
        
        return total_loss, regularization_loss, loc_loss, cls_loss, normalizer
           
def bn_relu(in_put, activation, is_training, scope):
    net = bn_(input_ = in_put, is_training = is_training, scope = scope)
    if activation:
        net = tf.nn.relu6(net)
    return net
    
def activa_conv_gn_reuse(in_put_, filters, kernel_size, strides, rate, scope, BN_scope, activation = True, is_training = True):
    net = conv_2d_same(in_put_, filters, kernel_size, strides, rate, is_training, scope)
    net = bn_(input_ = net, is_training = is_training, scope = BN_scope + scope)
    if activation:
        net = tf.nn.relu6(net)
    return net
        