# -*- coding: utf-8 -*-
import os
import numpy as np
import cv2
import pickle
from data import config as cfg
#import config as cfg
import time
import json

class ucf_data(object):
    def __init__(self,phase):

        if phase == 'train':
            self.batch_size = cfg.train_batch_size
            file_path = './data/ml_userInfo_train'
        else:
            self.batch_size = cfg.test_batch_size
            file_path = './data/ml_userInfo_test' 
        self.cursor = 0
        self.userInf = []
        self.vid2label = {}
        self.num_test = 0
        self.maxtime = 1427784002
        self.dtime = self.maxtime - 789652004
        vid2label = {}
        vidlist = []

        cn = 0
        with open('./data/ml_items') as f:
            for line in f:
                vid = line.split('\n')[0]
                vid = vid.split('\t')[0]
                vid2label[vid] = cn
                vidlist.append(vid)
                cn = cn+1
        self.vid2label = vid2label
        self.num_classes = len(self.vid2label)

        with open(file_path) as f:
            for line in f:
                line = line.split('\n')[0]
                data = json.loads(line)
                data_list = []
                time_list = []
                for item, stime in data.items():
                    data_list.append(item)
                    time_list.append(stime)
                if len(data_list)<10:
                    continue
                for i in range(len(data_list)-9):
                    tmp = {}
                    tmp['history'] = data_list[i:i+9]
                    tmp['obj'] = data_list[i+9]
                    tmp['ex_age'] = time_list[i+9]
                    self.userInf.append(tmp)
                self.num_test = self.num_test+1
        np.random.shuffle(self.userInf)  
        print('uderInf',len(self.userInf)) 
        #print('all_vec',len(self.vid2vec),vec.shape)
        self.num_classes = len(self.vid2label)

    def time_cal(self):
        time_now = int(time.time())
        time_local = time.localtime(time_now)
        dt = time.strftime("%Y-%m-%d %H:%M:%S",time_local)
        timeArray = time.strptime(dt, "%Y-%m-%d %H:%M:%S")
        timestamp = time.mktime(timeArray)
        return int(timestamp)

    def get(self):
        history = np.zeros((self.batch_size,9),np.int32)
        ex_age = np.zeros((self.batch_size,1),np.float32)
        labels = np.zeros((self.batch_size,1),np.int32)
        count = 0
        time_stamp = self.time_cal()
        while count < self.batch_size:
            data = self.userInf[self.cursor]
            history_vidlist = data['history']
            his_label_list = []
            for i in range(len(history_vidlist)):
                his_label_list.append(self.vid2label[history_vidlist[i]])
            history_label = np.array(his_label_list)
            tmp_label = self.vid2label[data['obj']]
            history[count, :] = history_label
            labels[count, :] = tmp_label
            ex_age[count, :] = self.maxtime - data['ex_age']
            #import pdb
            #pdb.set_trace()
            count = count+1
            self.cursor = self.cursor + 1
            
            if self.cursor >= len(self.userInf):
                np.random.shuffle(self.userInf)
                self.cursor = 0

        return history, ex_age, labels

def main():

    data = ucf_data('test')
    for i in range(100):
        start = time.time()
        history, ex_age, labels = data.get()
        t = time.time() - start
        print('history',history,history.shape)
        print('labels',labels,labels.shape)
        print('time',t)

if __name__ == '__main__':
    main()
            

        
