import tensorflow as tf
# def bn_()

def bn_(input_, esp=1e-3, is_training = True, decay = 0.99, scope = 'bn'):

    x = tf.layers.batch_normalization(
        inputs = input_,
        axis=-1,
        name = scope,
        momentum= 0.997,
        epsilon= 1e-4,
        training= is_training)
#         fused=True)
    return x
        
def gn_(input_, esp=1e-5, is_training = True, G=32, scope = 'gn'):
    with tf.variable_scope(scope):
        with tf.variable_scope('GroupNorm'):
            # normalize
            # tranpose: [bs, h, w, c] to [bs, c, h, w] following the paper
            x = input_
            x = tf.transpose(x, [0, 3, 1, 2])
            N, C, H, W = x.get_shape().as_list()
            G = min(G, C)
            x = tf.reshape(x, [-1, G, C // G, H, W])
            mean, var = tf.nn.moments(x, [2, 3, 4], keep_dims=True)
            x = (x - mean) / tf.sqrt(var + esp)
            # per channel gamma and beta

            gamma = tf.Variable(tf.constant(1.0, shape=[C]), dtype=tf.float32, name='gamma')
            beta = tf.Variable(tf.constant(0.0, shape=[C]), dtype=tf.float32, name='beta')
            gamma = tf.reshape(gamma, [1, C, 1, 1])
            beta = tf.reshape(beta, [1, C, 1, 1])

            output = tf.reshape(x, [-1, C, H, W]) * gamma + beta
            # tranpose: [bs, c, h, w, c] to [bs, h, w, c] following the paper
            output = tf.transpose(output, [0, 2, 3, 1])
    return output