import os
#os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import tensorflow as tf
import numpy as np
from data import config as cfg
from tensorflow.contrib import layers
slim = tf.contrib.slim


class video_model(object):

    def __init__(self,training):
        
        self.training = training
        self.batch_size = cfg.train_batch_size if self.training else cfg.test_batch_size
        self.kernel_regu = layers.l2_regularizer(0.001)
       
    def youtube_network(self, history, example_age, num_classes, labels= None):
        item_embedding = tf.get_variable('age_embedding',[num_classes,24],initializer=tf.variance_scaling_initializer())
        history_vec = tf.nn.embedding_lookup(item_embedding,history)
        print('history_vec',history_vec)
        inputs = tf.reduce_mean(history_vec,axis=1)
        print('history_vec_mean',inputs)
        #ex_age = tf.square(example_age)
        inputs = tf.concat([inputs, example_age],axis=1)
        print('inputs',inputs)
        
        net = tf.layers.dense(inputs,128,kernel_regularizer=self.kernel_regu,activation=tf.nn.relu,name='fc1') 
        net = tf.layers.batch_normalization(net,training = self.training,name = 'fc1_bn')
        net = tf.layers.dense(net,64,kernel_regularizer=self.kernel_regu,activation=tf.nn.relu,name='fc2')
        net = tf.layers.batch_normalization(net,training = self.training,name = 'fc2_bn')
        net = tf.layers.dense(net,24,kernel_regularizer=self.kernel_regu,activation=tf.nn.relu,name='fc3')
        net = tf.layers.batch_normalization(net,training = self.training,name = 'fc3_bn')
        weights = tf.get_variable('soft_weight',[num_classes,24],initializer=tf.variance_scaling_initializer())
        biases = tf.get_variable('soft_biases',initializer=tf.zeros([num_classes]),trainable=False)
        logits = tf.matmul(net, tf.transpose(weights))
        print('logits in network',logits)
        if self.training:
            with tf.device('/cpu:0'):
                losses = tf.reduce_mean(tf.nn.sampled_softmax_loss(weights, biases, labels, net,
                                               num_sampled=100, num_classes=num_classes, num_true=1,
                                               partition_strategy="div"))
        else:
            losses = None
        return net, logits, losses
            
def main():
    model = video_model(True)

if __name__ == '__main__':
    main()


        
