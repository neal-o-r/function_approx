import tensorflow as tf

def get_scope_variable(scope_name, var, shape=None, init=None):
    
        with tf.variable_scope(scope_name) as scope:
                try:
                        v = tf.get_variable(var, shape, initializer=init)
        
                except ValueError:
        
                        scope.reuse_variables()
                        v = tf.get_variable(var)
        
        return v


def feed_forward(neurons, n_next_layer, scope, name):
        
        assert 1 in neurons.get_shape()
        if 1 != neurons.get_shape()[0]:
                neurons = tf.transpose(neurons)
        
        w_init = tf.random_normal_initializer(stddev=0.1)
        b_init = tf.constant_initializer(0.0)

        w = get_scope_variable(scope, 'w'+name, [neurons.get_shape()[1], n_next_layer], w_init)
        b = get_scope_variable(scope, 'b'+name, [n_next_layer], b_init)
        
        return tf.matmul(neurons, w) + b

