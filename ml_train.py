import os
os.environ['CUDA_VISIBLE_DEVICES'] = '7'
import time
from data.get_ml import ucf_data  
import tensorflow as tf
import numpy as np
from model_ml import video_model
from data import config as cfg
slim = tf.contrib.slim


class Multi_Trainer(object):

    def __init__(self,model,data1):

        self.batch_size = cfg.train_batch_size
        self.model = model
        self.data1 = data1
        self.num_gpus = 1
        self.num_classes = data1.num_classes
        
    def tower_loss(self, history, ex_age, labels):
        net, logit, losses = self.model.youtube_network(history, ex_age,  self.num_classes,labels)
        regularization_losses = tf.reduce_mean(tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES))
        total_loss = losses + regularization_losses
        print(total_loss)
        return total_loss,logit

    def tower_acc(self, logit, labels):
        labels = tf.one_hot(labels,self.num_classes, axis=1)
        print('labels in acc',labels)
        correct_pred = tf.equal(tf.argmax(logit, 1), tf.argmax(labels,1))
        accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
        return accuracy

    def average_gradients(self, tower_grads):

        average_grads = []
        for grad_and_vars in zip(*tower_grads):
            grads = []
            for g, _ in grad_and_vars:
                expanded_g = tf.expand_dims(g, 0)
                grads.append(expanded_g)
            grad = tf.concat(grads, 0)
            grad = tf.reduce_mean(grad, 0)
            v = grad_and_vars[0][1]
            grad_and_var = (grad, v)
            average_grads.append(grad_and_var)
        return average_grads

    
    def train(self):

        with tf .Graph().as_default(), tf.device('/cpu:0'):

            global_step = tf.train.create_global_step()
            history_placeholder = tf.placeholder(tf.int32, (self.batch_size*self.num_gpus, 9))
            print('history',history_placeholder)
            example_age_placeholder = tf.placeholder(tf.float32, (self.batch_size*self.num_gpus, 1))
            label_placeholder = tf.placeholder(tf.int32, (self.batch_size*self.num_gpus, 1))
         
            tower_grads = []
            logits = []
            learning_rate = tf.train.exponential_decay(cfg.learning_rate,global_step,cfg.decay_steps,cfg.decay_rate,staircase = True)
            opt = tf.train.GradientDescentOptimizer(learning_rate)
            weight_file = None#'./checkpoint2/model.ckpt-855000'
            
            with tf.variable_scope(tf.get_variable_scope()):
                for gpu_index in range(0,self.num_gpus):
                    with tf.device('/gpu:%d' % gpu_index):
                        loss,logit = self.tower_loss(history_placeholder[gpu_index * self.batch_size:(gpu_index + 1) * self.batch_size, :],
                                                     example_age_placeholder[gpu_index * self.batch_size:(gpu_index + 1) * self.batch_size, :],
                                                     label_placeholder[gpu_index * self.batch_size:(gpu_index + 1) * self.batch_size, :])
                        tf.get_variable_scope().reuse_variables()
                        batchnorm_updates = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
                        grads1 = opt.compute_gradients(loss)
                        tower_grads.append(grads1)
                        logits.append(logit)

            logits = tf.concat(logits, 0)
            accuracy = self.tower_acc(logits, label_placeholder)
            grads = self.average_gradients(tower_grads)
            apply_gradient_op = opt.apply_gradients(grads)

            variable_averages = tf.train.ExponentialMovingAverage(0.99, global_step)
            variables_to_average = (tf.trainable_variables() + tf.moving_average_variables())
            variables_averages_op = variable_averages.apply(variables_to_average)
            batchnorm_updates_op = tf.group(*batchnorm_updates)
            train_op = tf.group(apply_gradient_op, variables_averages_op,batchnorm_updates_op)

            var_list = tf.trainable_variables()
            g_list = tf.global_variables()
            bn_moving_vars = [g for g in g_list if 'moving_mean' in g.name]
        
            bn_moving_vars += [g for g in g_list if 'moving_variance' in g.name]
            #print(bn_moving_vars)
            var_list += bn_moving_vars
            saver = tf.train.Saver(var_list,max_to_keep = 20)
            init = tf.global_variables_initializer()
            config = tf.ConfigProto()
            config.gpu_options.allow_growth = True
            sess = tf.Session()
            sess.run(init)
            if weight_file is not None:
                print('Restore weight file')
                saver.restore(sess,weight_file)

            coord = tf.train.Coordinator()
            threads = tf.train.start_queue_runners(sess=sess, coord=coord)
            f = open('result.txt','w+')
            print('Starting Training')
            start_time = time.time()

            for step in range(cfg.max_iter + 5):
                
                history, ex_age, label = self.data1.get()
                feed_dict = {history_placeholder: history,
                             example_age_placeholder: ex_age,
                             label_placeholder: label}

     
                _,acc,loss_value,gstep = sess.run([train_op,accuracy,loss,global_step], feed_dict=feed_dict)
                
                gstep = int(gstep)
                if step % 100 ==0:

                    duration = time.time() - start_time
                    print('step: %d  loss: %.5f  acc: %.5f  time:%.5f'%(step,loss_value,acc,duration))
                    start_time = time.time()

                if step % 5000 == 0 :
                    '''                   
                    acc_value = 0
                    for i in range(15):

                        history,age,sex,example_age,label = self.data2.get()
                        feed_dict = {history_placeholder:history,
                                     age_placeholder:age,
                                     sex_placeholder:sex,
                                     example_age_placeholder:example_age,
                                     label_placeholder:label}
                        acc = sess.run(accuracy,feed_dict = feed_dict)
                        acc_value = acc_value + acc
                    acc_value = acc_value / 15.0
                    print('After %d steps, the test accuracy is: %.5f'%(step,acc_value))
                    str1 = 'steps:' + '\t' + str(step) + '\t' + 'accuracy:' + '\t' + str(acc_value) + '\n'
                    f.write(str1)
                    '''
                    saver.save(sess,'checkpoint1/model.ckpt',global_step = step)

            coord.request_stop()
            coord.join(threads)
            print('Ending Training')
            f.close()

def main():
    net = video_model(True)
    data1 = ucf_data('train')
    #data2 = ucf_data('test')
    Trainer = Multi_Trainer(net,data1)
    Trainer.train()

if __name__ == '__main__':
    
    main()


                


                

            
                   






            

            
