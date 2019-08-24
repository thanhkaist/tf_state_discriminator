import numpy as np
import tensorflow as tf
import time
import inspect
import os
import tensorflow as tf
from tensorflow.contrib.layers import flatten

class Vgg16:
    def __init__(self, vgg16_npy_path=None):
        if vgg16_npy_path is None:
            path = inspect.getfile(Vgg16)
            path = os.path.abspath(os.path.join(path, os.pardir))
            path = os.path.join(path, "vgg16.npy")
            vgg16_npy_path = path
            print(path)

        self.data_dict = np.load(vgg16_npy_path, encoding='latin1').item()
        print("npy file loaded")

    def build(self,s_img,t_img):
        """
        load variable from npy to build the VGG
	    :param s_img: rgb image [batch, height, width, 3] values scaled [0, 255]
	    :param rgb: rgb image [batch, height, width, 3] values scaled [0, 255]
	    """

        start_time = time.time()
        print("build model started")
        s_rgb_scaled = s_img
	    t_rgb_scaled = t_img

        # Convert RGB to BGR
        red, green, blue = tf.split(axis=3, num_or_size_splits=3, value=s_rgb_scaled)
	    red1, green1, blue1 = tf.split(axis=3, num_or_size_splits=3, value=t_rgb_scaled)
	# check size if it is not correct        
	    assert red.get_shape().as_list()[1:] == [224, 224, 1]
        assert green.get_shape().as_list()[1:] == [224, 224, 1]
        assert blue.get_shape().as_list()[1:] == [224, 224, 1]
	    assert red1.get_shape().as_list()[1:] == [224, 224, 1]
        assert green1.get_shape().as_list()[1:] == [224, 224, 1]
        assert blue1.get_shape().as_list()[1:] == [224, 224, 1]

        bgr = tf.concat(axis=3, values=[
            blue - VGG_MEAN[0],
            green - VGG_MEAN[1],
            red - VGG_MEAN[2],
            blue1 - VGG_MEAN[0],
            green1 - VGG_MEAN[1],
            red1 - VGG_MEAN[2],
        ])
        assert bgr.get_shape().as_list()[1:] == [224, 224, 6]

        self.conv1_1 = self.conv_layer(bgr, "conv1_1")
        self.conv1_2 = self.conv_layer(self.conv1_1, "conv1_2")
        self.pool1 = self.max_pool(self.conv1_2, 'pool1')

        self.conv2_1 = self.conv_layer(self.pool1, "conv2_1")
        self.conv2_2 = self.conv_layer(self.conv2_1, "conv2_2")
        self.pool2 = self.max_pool(self.conv2_2, 'pool2')

        self.conv3_1 = self.conv_layer(self.pool2, "conv3_1")
        self.conv3_2 = self.conv_layer(self.conv3_1, "conv3_2")
        self.conv3_3 = self.conv_layer(self.conv3_2, "conv3_3")
        self.pool3 = self.max_pool(self.conv3_3, 'pool3')

        self.conv4_1 = self.conv_layer(self.pool3, "conv4_1")
        self.conv4_2 = self.conv_layer(self.conv4_1, "conv4_2")
        self.conv4_3 = self.conv_layer(self.conv4_2, "conv4_3")
        self.pool4 = self.max_pool(self.conv4_3, 'pool4')

        self.conv5_1 = self.conv_layer(self.pool4, "conv5_1")
        self.conv5_2 = self.conv_layer(self.conv5_1, "conv5_2")
        self.conv5_3 = self.conv_layer(self.conv5_2, "conv5_3")
        self.pool5 = self.max_pool(self.conv5_3, 'pool5')

        self.fc6 = self.fc_layer(self.pool5, "fc6")
        assert self.fc6.get_shape().as_list()[1:] == [4096]
        self.relu6 = tf.nn.relu(self.fc6)

        self.fc7 = self.fc_layer(self.relu6, "fc7")
        self.relu7 = tf.nn.relu(self.fc7)

        self.fc8 = self.fc_layer(self.relu7, "fc8")

        self.prob = tf.nn.softmax(self.fc8, name="prob")

        self.data_dict = None
        print(("build model finished: %ds" % (time.time() - start_time)))

    def avg_pool(self, bottom, name):
        return tf.nn.avg_pool(bottom, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name=name)

    def max_pool(self, bottom, name):
        return tf.nn.max_pool(bottom, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name=name)

    def conv_layer(self, bottom, name):
        with tf.variable_scope(name):
            filt = self.get_conv_filter(name)

            conv = tf.nn.conv2d(bottom, filt, [1, 1, 1, 1], padding='SAME')

            conv_biases = self.get_bias(name)
            bias = tf.nn.bias_add(conv, conv_biases)

            relu = tf.nn.relu(bias)
            return relu

    def fc_layer(self, bottom, name):
        with tf.variable_scope(name):
            shape = bottom.get_shape().as_list()
            dim = 1
            for d in shape[1:]:
                dim *= d
            x = tf.reshape(bottom, [-1, dim])

            weights = self.get_fc_weight(name)
            biases = self.get_bias(name)

            # Fully connected layer. Note that the '+' operation automatically
            # broadcasts the biases.
            fc = tf.nn.bias_add(tf.matmul(x, weights), biases)

            return fc

    def get_conv_filter(self, name):
        return tf.constant(self.data_dict[name][0], name="filter")

    def get_bias(self, name):
        return tf.constant(self.data_dict[name][1], name="biases")

    def get_fc_weight(self, name):
        return tf.constant(self.data_dict[name][0], name="weights")


def mlp(x, hidden_sizes=(32,), activation=tf.tanh, output_activation=None):
    for h in hidden_sizes[:-1]:
        x = tf.layers.dense(x, units=h, activation=activation)
    return tf.layers.dense(x, units=hidden_sizes[-1], activation=output_activation)


def init_weight(shape):
    w = tf.truncated_normal(shape=shape, mean = 0, stddev = 0.1)
    return tf.Variable(w,name="weights")
    
def init_bias(shape):
    b = tf.zeros(shape)
    return tf.Variable(b,name="bias")

def TargetDiscriminator(s_img, t_img):
    """
    load variable from npy to build the VGG
    :param s_img: rgb image [batch, height, width, 3] values scaled [0, 255]
    :param rgb: rgb image [batch, height, width, 3] values scaled [0, 255]
    """

    start_time = time.time()
    print("build model started")
    s_rgb_scaled = s_img
    t_rgb_scaled = t_img

    # Convert RGB to BGR
    red, green, blue = tf.split(axis=3, num_or_size_splits=3, value=s_rgb_scaled)
    red1, green1, blue1 = tf.split(axis=3, num_or_size_splits=3, value=t_rgb_scaled)
    # check size if it is not correct
    assert red.get_shape().as_list()[1:] == [224, 224, 1]
    assert green.get_shape().as_list()[1:] == [224, 224, 1]
    assert blue.get_shape().as_list()[1:] == [224, 224, 1]
    assert red1.get_shape().as_list()[1:] == [224, 224, 1]
    assert green1.get_shape().as_list()[1:] == [224, 224, 1]
    assert blue1.get_shape().as_list()[1:] == [224, 224, 1]

    bgr = tf.concat(axis=3, values=[
        blue - VGG_MEAN[0],
        green - VGG_MEAN[1],
        red - VGG_MEAN[2],
        blue1 - VGG_MEAN[0],
        green1 - VGG_MEAN[1],
        red1 - VGG_MEAN[2],
    ])
    assert bgr.get_shape().as_list()[1:] == [224, 224, 6]

    # name:      conv5-6    
    # structure: Input = 224x224x6. Output = 220x220x6.
    # weights:   (5*5*1+1)*6
    # connections: (28*28*5*5+28*28)*6


    with tf.variable_scope("conv5-6"):
        conv1_W = init_weight((5,5,6,6))
        conv1_b = init_bias(6)
        conv1   = tf.nn.conv2d(x, conv1_W, strides=[1, 1, 1, 1], padding='VALID') + conv1_b
        conv1 = tf.nn.relu(conv1)
    #Input = 220x220x6. Output = 110x110x6.
    conv1 = tf.nn.max_pool(conv1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')
    
    #conv5-16
    #input 14x14x6 Output = 10x10x16.
    #weights: (5*5*6+1)*16 ---real Lenet-5 is (5*5*3+1)*6+(5*5*4+1)*9+(5*5*6+1)*1
    with tf.variable_scope("conv5-16"):
        conv2_W = init_weight((5, 5, 6, 16))
        conv2_b = init_bias(16)
        conv2   = tf.nn.conv2d(conv1, conv2_W, strides=[1, 1, 1, 1], padding='VALID') + conv2_b
        conv2 = tf.nn.relu(conv2)
    #Input = 10x10x16. Output = 5x5x16.
    conv2 = tf.nn.max_pool(conv2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')
    #Input = 5x5x16. Output = 400.
    fc0   = flatten(conv2)
    
    #Input = 400. Output = 120.
    fc1_W = init_weight((400,120))
    fc1_b = init_bias(120)
    fc1   = tf.matmul(fc0, fc1_W) + fc1_b
    fc1    = tf.nn.relu(fc1)

    #Input = 120. Output = 84.
    fc2_W  = init_weight((120,84))
    fc2_b  = init_bias(84)
    fc2    = tf.matmul(fc1, fc2_W) + fc2_b
    fc2 = tf.nn.relu(fc2)

    #Input = 84. Output = 10.
    fc3_W  = init_weight((84,10))
    fc3_b  = init_bias(10)
    logits = tf.matmul(fc2, fc3_W) + fc3_b
    
    return logits



def NeighborDiscriminator(img1,img2):
	




# Simple script for training an MLP on MNIST.
def train_mnist(steps_per_epoch=100, epochs=5, 
                lr=1e-3, layers=2, hidden_size=64, 
                logger_kwargs=dict(), save_freq=1):

    logger = EpochLogger(**logger_kwargs)
    logger.save_config(locals())

    # Load and preprocess MNIST data
    (x_train, y_train), _ = tf.keras.datasets.mnist.load_data()
    x_train = x_train.reshape(-1, 28*28) / 255.0

    # Define inputs & main outputs from computation graph
    x_ph = tf.placeholder(tf.float32, shape=(None, 28*28))
    y_ph = tf.placeholder(tf.int32, shape=(None,))
    logits = mlp(x_ph, hidden_sizes=[hidden_size]*layers + [10], activation=tf.nn.relu)
    predict = tf.argmax(logits, axis=1, output_type=tf.int32)

    # Define loss function, accuracy, and training op
    y = tf.one_hot(y_ph, 10)
    loss = tf.losses.softmax_cross_entropy(y, logits)
    acc = tf.reduce_mean(tf.cast(tf.equal(y_ph, predict), tf.float32))
    train_op = tf.train.AdamOptimizer().minimize(loss)

    # Prepare session
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    # Setup model saving
    logger.setup_tf_saver(sess, inputs={'x': x_ph}, 
                                outputs={'logits': logits, 'predict': predict})

    start_time = time.time()

    # Run main training loop
    for epoch in range(epochs):
        for t in range(steps_per_epoch):
            idxs = np.random.randint(0, len(x_train), 32)
            feed_dict = {x_ph: x_train[idxs],
                         y_ph: y_train[idxs]}
            outs = sess.run([loss, acc, train_op], feed_dict=feed_dict)
            logger.store(Loss=outs[0], Acc=outs[1])

        # Save model
        if (epoch % save_freq == 0) or (epoch == epochs-1):
            logger.save_state(state_dict=dict(), itr=None)

        # Log info about epoch
        logger.log_tabular('Epoch', epoch)
        logger.log_tabular('Acc', with_min_and_max=True)
        logger.log_tabular('Loss', average_only=True)
        logger.log_tabular('TotalGradientSteps', (epoch+1)*steps_per_epoch)
        logger.log_tabular('Time', time.time()-start_time)
        logger.dump_tabular()

if __name__ == '__main__':
    train_mnist()
