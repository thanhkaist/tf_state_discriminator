from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='3'
import tensorflow as tf
import time
import numpy as np

def preprocess_data(im1,im2,label):
    im1=tf.cast(im1,tf.float32)
    im2 =tf.cast(im2, tf.float32)
    label=tf.cast(label, tf.int32)
    #im1 =im1/127.5
    #im1 = im1 - 1
    #im2 =im2/127.5
    #im2 = im2 - 1
    return im1,im2,label

def load_data_from_npz(no_train=800,no_test=200, path='pair_obs.npz'):
    data = np.load(path)
    pair_obs_arr = data['pair_obs_arr']
    labels = data['labels']
    len = pair_obs_arr.shape[0]
    if len < (no_test +no_train):
        raise ValueError("Len of data less than toal no_train + no_test: %1".format(len))


    img1s = pair_obs_arr[:no_train,0]
    img2s = pair_obs_arr[:no_train,1]
    labels_t = labels[:no_train]
    img1s_test =pair_obs_arr[no_train:no_train+no_test,0]
    img2s_test =pair_obs_arr[no_train:no_train+no_test,1]
    labels_test = labels[no_train:no_train+no_test]
    print(img1s_test.shape,img2s_test.shape,labels_test.shape)
    train_data = (img1s,img2s,labels_t)
    test_data = (img1s_test, img2s_test,labels_test)

    return  train_data, test_data

def create_dataset_pipeline(data_tensor, is_train=True, num_threads=8, prefetch_buffer=100, batch_size=32):
    dataset = tf.data.Dataset.from_tensor_slices(data_tensor)
    if is_train:
        dataset = dataset.shuffle(buffer_size=800).repeat()
    dataset = dataset.map(preprocess_data, num_parallel_calls=num_threads)
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(prefetch_buffer)
    return dataset

def data_layer():
    with tf.variable_scope("data"):
        #data_train, data_val = tf.keras.datasets.mnist.load_data()
        data_train, data_val = load_data_from_npz(800,200,"Data/pair_obs.npz")
        dataset_train = create_dataset_pipeline(data_train, is_train=True)
        dataset_val = create_dataset_pipeline(data_val, is_train=False, batch_size=1)
        iterator = tf.data.Iterator.from_structure(dataset_train.output_types, dataset_train.output_shapes)
        init_op_train = iterator.make_initializer(dataset_train)
        init_op_val = iterator.make_initializer(dataset_val)
    return iterator, init_op_train, init_op_val

def init_weight(shape):
    w = tf.truncated_normal(shape=shape,mean = 0, stddev = 0.1)
    return tf.Variable(w, name = 'weight')

def init_bias(shape):
    b = tf.zeros(shape)
    return tf.Variable(b, name='bias')

def target_discrim_model(s_img,t_img):
    '''
    Generate the model graph
    :param s_img: state [1,224,224,3]
    :param t_img: goal state [1,224,224,3]
    :return: model of discrim with two output tensor
    '''
    with tf.variable_scope("target"):
        start_time = time.time()
        print("build model started")
        #rgb_scaled = rgb * 255.0

        # Convert RGB to BGR
        red, green, blue = tf.split(axis=3, num_or_size_splits=3, value=s_img)
        red1, green1, blue1 = tf.split(axis=3, num_or_size_splits=3, value=t_img)
        assert red.get_shape().as_list()[1:] == [224, 224, 1]
        assert green.get_shape().as_list()[1:] == [224, 224, 1]
        assert blue.get_shape().as_list()[1:] == [224, 224, 1]
        assert red1.get_shape().as_list()[1:] == [224, 224, 1]
        assert green1.get_shape().as_list()[1:] == [224, 224, 1]
        assert blue1.get_shape().as_list()[1:] == [224, 224, 1]
        bgr = tf.concat(axis=3, values=[
            blue ,
            green,
            red ,
            blue1,
            green1,
            red1,
        ])
        assert bgr.get_shape().as_list()[1:] == [224, 224, 6]
        with tf.variable_scope("conv5-6"):
            # name:      conv5-6
            # structure: Input = 224x224x6. Output = 220x220x6.
            # weights:   (5*5*6+1)*6
            # connections: (28*28*5*5+28*28)*6
            conv1_W = init_weight((5, 5, 6, 6))
            conv1_b = init_bias(6)
            conv1 = tf.nn.conv2d(bgr, conv1_W, strides=[1, 1, 1, 1], padding='VALID') + conv1_b
            conv1 = tf.nn.relu(conv1)
        # Input = 220x220x6. Output = 110x110x6.
        conv1 = tf.nn.max_pool(conv1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')
        with tf.variable_scope("conv3-16"):
            # conv3-16
            # input 110x110x6 Output = 108x108x16.
            # weights: (5*5*6+1)*16
            conv2_W = init_weight((3, 3, 6, 16))
            conv2_b = init_bias(16)
            conv2 = tf.nn.conv2d(conv1, conv2_W, strides=[1, 1, 1, 1], padding='VALID') + conv2_b
            conv2 = tf.nn.relu(conv2)
        # Input = 108x108x16. Output = 54x54x16.
        conv2 = tf.nn.max_pool(conv2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')
        with tf.variable_scope("conv3-8"):
            # conv3-8
            # input 54x54x16 Output = 52x52x8.
            # weights: (3*3*16+1)*16

            conv3_W = init_weight((3, 3, 16, 8))
            conv3_b = init_bias(8)
            conv3 = tf.nn.conv2d(conv2, conv3_W, strides=[1, 1, 1, 1], padding='VALID') + conv3_b
            conv3 = tf.nn.relu(conv3)
        # Input = 52x52x16. Output = 26*26*8
        conv3 = tf.nn.max_pool(conv3, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')

        with tf.variable_scope("fc"):
            # Input = 26*26*8. Output = 5408
            net = tf.layers.flatten(conv3)
            assert net.get_shape().as_list()[1:] == [5408]
            net = tf.layers.dense(net, 1024)
            net = tf.nn.relu(net)
            net = tf.layers.dense(net, 500)
            net = tf.nn.relu(net)
            net = tf.layers.dense(net, 2)
        print(("build model finished: %ds" % (time.time() - start_time)))
        return net

def loss_functions(logits, labels, num_classes=2):
    with tf.variable_scope("loss"):
        target_prob = tf.one_hot(labels, num_classes)
        tf.losses.softmax_cross_entropy(target_prob, logits)
        total_loss = tf.losses.get_total_loss() # include regularization loss
    return total_loss

def optimizer_func_momentum(total_loss, global_step, learning_rate=0.01):
    with tf.variable_scope("optimizer"):
        lr_schedule = tf.train.exponential_decay(learning_rate=learning_rate,
                                                 global_step=global_step,
                                                 decay_steps=25*5,
                                                 decay_rate=0.5,
                                                 staircase=True)
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            optimizer = tf.train.MomentumOptimizer(learning_rate=lr_schedule, momentum=0.9)
            optimizer = optimizer.minimize(total_loss, global_step=global_step)
    return optimizer

def optimizer_func_adam(total_loss, global_step, learning_rate=0.01):
    with tf.variable_scope("optimizer"):
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate, epsilon=0.1)
            optimizer = optimizer.minimize(total_loss, global_step=global_step)
    return optimizer

def performance_metric(logits, labels):
    with tf.variable_scope("performance_metric"):
        preds = tf.argmax(logits, axis=1)
        labels = tf.cast(labels, tf.int64)
        corrects = tf.equal(preds, labels)
        accuracy = tf.reduce_mean(tf.cast(corrects, tf.float32))
    return accuracy


def train():
    global_step = tf.Variable(1,dtype=tf.int32,trainable=False,name='iter_number')

    # defind the training graph
    # s_img = tf.placeholder(tf.float32,shape=(None,224,224,3))
    # t_img = tf.placeholder(tf.float32,shape=(None,224,224,3))
    # labels = tf.placeholder(tf.int32,shape=(None,))
    # logits = target_discrim_model(s_img,t_img)

    iterator, init_op_train, init_op_val = data_layer()
    s_img,t_img,labels = iterator.get_next()

    logits = target_discrim_model(s_img, t_img)

    loss = loss_functions(logits,labels,num_classes=2)
    #optimizer = optimizer_func_adam(loss,global_step)
    optimizer = optimizer_func_momentum(loss,global_step)
    accuracy = performance_metric(logits,labels)

    # summary placeholders
    streaming_loss_p = tf.placeholder(tf.float32)
    streaming_acc_p = tf.placeholder(tf.float32)
    val_acc_p = tf.placeholder(tf.float32)
    val_summ_ops = tf.summary.scalar('validation_acc', val_acc_p)
    train_summ_ops = tf.summary.merge([
        tf.summary.scalar('streaming_loss', streaming_loss_p),
        tf.summary.scalar('streaming_accuracy', streaming_acc_p)
    ])

    # start training
    num_iter = 800*10//32 # 100 epochs
    log_iter = 200//32#1875
    val_iter = 200//32#1875
    training = tf.placeholder(tf.bool)
    # fake data

    #img1 = np.random.randint(0,255,(5,224,224,3))  #tf.random.uniform((5,224,224,3),0,255)
    #img2 = np.random.randint(0,255,(5,224,224,3))   #tf.random.uniform((5,224,224,3),0,255)
    #ex_labls = np.random.randint(0,2,(5,)) #tf.random.uniform((5,),0,2,tf.int32)


    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        sess.run(init_op_train)

        # logs for TensorBoard
        logdir = 'logs/target'
        writer = tf.summary.FileWriter(logdir, sess.graph) # visualize the graph

        # load / save checkpoints
        checkpoint_path = 'checkpoints/target'
        saver = tf.train.Saver(max_to_keep=None)
        ckpt = tf.train.get_checkpoint_state(checkpoint_path)

        # resume training if a checkpoint exists
        if ckpt and ckpt.model_checkpoint_path:
            saver.restore(sess, ckpt.model_checkpoint_path)
            print("Loaded parameters from {}".format(ckpt.model_checkpoint_path))

        initial_step = global_step.eval()

        streaming_loss = 0
        streaming_accuracy = 0

        for i in range(initial_step, num_iter + 1):
            # _, loss_batch, acc_batch = sess.run([optimizer, loss, accuracy], feed_dict={s_img:img1,t_img:img2,labels:ex_labls,training: True}) ##############################
            _, loss_batch, acc_batch = sess.run([optimizer, loss, accuracy], feed_dict={training: True})
            streaming_loss += loss_batch
            streaming_accuracy += acc_batch
            if i % log_iter == 0:
                print("Iteration: {}, Streaming loss: {:.6f}, Streaming accuracy: {:.6f}"
                        .format(i, streaming_loss/log_iter, streaming_accuracy/log_iter))

                # save to log file for TensorBoard
                summary_train = sess.run(train_summ_ops, feed_dict={streaming_loss_p: streaming_loss,
                                                                    streaming_acc_p: streaming_accuracy})
                writer.add_summary(summary_train, global_step=i)

                streaming_loss = 0
                streaming_accuracy = 0

            if i % val_iter == 0:
                saver.save(sess, os.path.join(checkpoint_path, 'checkpoint'), global_step=global_step)
                print("Model saved!")

                sess.run(init_op_val)
                validation_accuracy = 0
                num_iter = 0
                # import pdb
                # pdb.set_trace()
                while True:
                    try:
                        # acc_batch = sess.run(accuracy, feed_dict={s_img:img1,t_img:img2,labels:ex_labls,training: False}) ##############################
                        acc_batch = sess.run(accuracy, feed_dict={training: False})  ##############################
                        validation_accuracy += acc_batch
                        num_iter += 1

                        ## Thanh : test code
                        # validation_accuracy /= num_iter
                        # print("Iteration: {}, Validation accuracy: {:.2f}".format(i, validation_accuracy))
                        #
                        # # save log file to TensorBoard
                        # summary_val = sess.run(val_summ_ops, feed_dict={val_acc_p: validation_accuracy})
                        # writer.add_summary(summary_val, global_step=i)
                        #
                        # # sess.run(init_op_train) # switch back to training set
                        # break

                    except tf.errors.OutOfRangeError:
                        validation_accuracy /= num_iter
                        print("Iteration: {}, Validation accuracy: {:.2f}".format(i, validation_accuracy))

                        # save log file to TensorBoard
                        summary_val = sess.run(val_summ_ops, feed_dict={val_acc_p: validation_accuracy})
                        writer.add_summary(summary_val, global_step=i)

                        sess.run(init_op_train) # switch back to training set
                        break
        writer.close()

def test(no_checkpoint):
    global_step = tf.Variable(1,dtype=tf.int32,trainable=False,name='iter_number')

    iterator, init_op_train, init_op_val = data_layer()
    s_img,t_img,labels = iterator.get_next()

    logits = target_discrim_model(s_img, t_img)

    loss = loss_functions(logits,labels,num_classes=2)
    optimizer = optimizer_func_momentum(loss,global_step)
    accuracy = performance_metric(logits,labels)

    # summary placeholders
    streaming_loss_p = tf.placeholder(tf.float32)
    streaming_acc_p = tf.placeholder(tf.float32)
    val_acc_p = tf.placeholder(tf.float32)
    val_summ_ops = tf.summary.scalar('validation_acc', val_acc_p)
    train_summ_ops = tf.summary.merge([
        tf.summary.scalar('streaming_loss', streaming_loss_p),
        tf.summary.scalar('streaming_accuracy', streaming_acc_p)
    ])

    # start training
    num_iter = 800*10//32 # 100 epochs
    log_iter = 200//32#1875
    val_iter = 200//32#1875
    training = tf.placeholder(tf.bool)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        sess.run(init_op_train)

        # logs for TensorBoard
        logdir = 'logs/target'

        # load / save checkpoints
        checkpoint_path = 'checkpoints/target'
        saver = tf.train.Saver(max_to_keep=None)
        ckpt = tf.train.get_checkpoint_state(checkpoint_path)
        # import pdb
        #  pdb.set_trace()

        # resume training if a checkpoint exists
        if ckpt and ckpt.model_checkpoint_path:
            saver.restore(sess, ckpt.model_checkpoint_path)
            print("Loaded parameters from {}".format(ckpt.model_checkpoint_path))

        initial_step = global_step.eval()
        sess.run(init_op_val)
        validation_accuracy = 0
        num_iter = 0
        while True:
            try:
                acc_batch = sess.run(accuracy, feed_dict={training: False})  ##############################
                validation_accuracy += acc_batch
                num_iter += 1


            except tf.errors.OutOfRangeError:
                validation_accuracy /= num_iter
                print("Validation accuracy: {:.2f}".format(validation_accuracy))
                break

                # save log file to TensorBoard


if __name__ == "__main__":
    #train()
    test(10)
