TensorFlow implementation of the paper "Deep Neural Networks for YouTube Recommendations"



Data prepare

Dataset:MovieLens-20M
data are prepared this way:

user id1:{"timestamp1":movie id1,"timestamp2": movie id2....}

user id2:{"timestamp1":movie id1,"timestamp2": movie id2....}

from the first watch history to the last history in time sort



Feature used:

watched history, example age

Run:

python ml_train.py

