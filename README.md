# TensorFlow implementation of the paper "Deep Neural Networks for YouTube Recommendations"



Data prepare

Dataset:MovieLens-20M
data are prepared this way:

user id1:{"timestamp1":movie id1,"timestamp2": movie id2....}

user id2:{"timestamp1":movie id1,"timestamp2": movie id2....}

* sorted in time from the first watch history to the last history


Feature used:

watched history, example age

Also, other demography features can be added, as long as the featrues are feeded into correct placeholder

### Run:

##### python ml_train.py

This code is recommended to run on only one GPU card. 

Howevre, it supports multiple GPU cards training but can not lift up training speed LOL.
