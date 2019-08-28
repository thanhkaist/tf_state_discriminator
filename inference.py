""" Runs inference given a frozen model and a set of images
Example:
$ python inference.py --frozen_model frozen_model.pb --input_path ./test_images
"""

import argparse
import tensorflow as tf
import os, glob
import cv2
import numpy as np 
os.environ['TF_CPP_MIN_LOG_LEVEL']='3'

def load_data_from_npz(path):
    data = np.load(path)
    pair_o = data['pair_goal']
    labels = data['labels']
    img1s = pair_o[:,0]
    img2s = pair_o[:,1]
    return img1s ,img2s,labels
class InferenceEngine:
    def __init__(self, frozen_graph_filename):
        with tf.gfile.GFile(frozen_graph_filename, "rb") as f:
            graph_def = tf.GraphDef()
            graph_def.ParseFromString(f.read())

        with tf.Graph().as_default() as graph:
            tf.import_graph_def(graph_def, name="Pretrained")

        self.graph = graph

    def run_inference(self, img_s,img_t):
        img_s_ph = self.graph.get_tensor_by_name('Pretrained/state_images:0')
        img_t_ph = self.graph.get_tensor_by_name('Pretrained/target_images:0')
        preds = self.graph.get_tensor_by_name('Pretrained/preds:0')
        pred_idx = tf.argmax(preds)

        with tf.Session(graph=self.graph) as sess:
            class_label, probs = sess.run([pred_idx, preds], feed_dict={img_s_ph:img_s,img_t_ph:img_t})
            # print("Label: {:d}, Probability: {:.2f} \t ".format(class_label, probs[class_label]))
        return class_label, probs[class_label]

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--frozen_model", default='./target.pd', type=str, help="Path to the frozen model file to import")
    parser.add_argument("--input_path", type=str,default = 'Data/pair_goal.npz', help="Path to the npz data")
    parser.add_argument("--index",type = int , help ="predicting data index",default=0)
    args = parser.parse_args()
    img1s ,img2s ,labels= load_data_from_npz(args.input_path)
    ie = InferenceEngine(args.frozen_model)
    #  predict curtain index
    if False:
        img1 = np.expand_dims(img1s[args.index],axis=0)
        img2 = np.expand_dims(img2s[args.index],axis=0)
        print('Actual label:', labels[args.index])
        ie.run_inference(img1,img2)
    else:
    # find wrong label 
        for i in range(1000):
            img1 = np.expand_dims(img1s[i],axis=0)
            img2 = np.expand_dims(img2s[i],axis=0)
            pre_label , prob = ie.run_inference(img1,img2)
            if  pre_label != labels[i]:
                print("Prediction is wrong at index {:d} with predict label {:d},wrong probability {:2f}".format(i,pre_label,prob))

