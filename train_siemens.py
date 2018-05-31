from __future__ import division
from __future__ import print_function

import time
import os
import sys

import tensorflow as tf
import numpy as np
import scipy.sparse as sp
import scipy.stats as stats
import networkx as nx

from sklearn.metrics import roc_auc_score
from sklearn.metrics import average_precision_score
from sklearn.preprocessing import normalize

import matplotlib.pyplot as plt

from sklearn import manifold
from scipy.special import expit

from optimizer import *
from input_data import *
from model import *
from preprocessing import *


# Settings
flags = tf.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_float('learning_rate', 0.01, 'Initial learning rate.')
flags.DEFINE_integer('epochs', 200, 'Number of epochs to train.')
flags.DEFINE_integer('hidden1', 32, 'Number of units in hidden layer 1.')
flags.DEFINE_integer('hidden2', 16, 'Number of units in hidden layer 2.')
flags.DEFINE_integer('hidden3', 32, 'Number of units in hidden layer 3.')
flags.DEFINE_integer('hidden4', 16, 'Number of units in hidden layer 4.')
flags.DEFINE_float('dropout', 0., 'Dropout rate (1 - keep probability).')
flags.DEFINE_float('edge_dropout', 0., 'Dropout for individual edges in training graph')
flags.DEFINE_float('autoregressive_scalar', 0., 'Scalar for Graphites')
flags.DEFINE_integer('vae', 1, '1 for variational objective')

flags.DEFINE_integer('subsample', 0, 'Subsample in optimizer')
flags.DEFINE_float('subsample_frac', 1, 'Ratio of sampled non-edges to edges if using subsampling')

flags.DEFINE_integer('verbose', 1, 'verboseness')
flags.DEFINE_integer('test_count', 10, 'batch of tests')

flags.DEFINE_integer('gpu', -1, 'Which gpu to use')
flags.DEFINE_integer('seeded', 1, 'Set numpy random seed')
np.set_printoptions(suppress=True, precision=3)

if FLAGS.seeded:
    np.random.seed(1)

A_orig, A, X = load_siemens()

features = X[0]
num_features = 245 + 1
features_nonzero = 245 * 2


placeholders = {
    'features': tf.sparse_placeholder(tf.float32),
    'adj': tf.sparse_placeholder(tf.float32),
    'adj_orig': tf.sparse_placeholder(tf.float32),
    'dropout': tf.placeholder_with_default(0., shape=()),
}

num_nodes = 245

model = GCNModelSiemens(placeholders, num_features, num_nodes, features_nonzero)




with tf.name_scope('optimizer'):
    opt = OptimizerSiemens(preds=model.reconstructions,
                       labels=tf.reshape(tf.sparse_tensor_to_dense(placeholders['adj_orig'], validate_indices=False), [-1]),
                       model=model, num_nodes=num_nodes,
                       pos_weight=1,
                       norm=1)

os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
if FLAGS.gpu == -1:
    os.environ['CUDA_VISIBLE_DEVICES'] = ''
    sess = tf.Session()
else:
    os.environ['CUDA_VISIBLE_DEVICES'] = str(FLAGS.gpu) # Or whichever device you would like to use
    gpu_options = tf.GPUOptions(allow_growth=True)
    sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options, allow_soft_placement=True))
sess.run(tf.global_variables_initializer())

for epoch in range(FLAGS.epochs):

    index = epoch % len(A)

    feed_dict = construct_feed_dict(A[index], A_orig[index], X[index], placeholders)
    feed_dict.update({placeholders['dropout']: FLAGS.dropout})
    outs = sess.run([opt.accuracy, opt.cost, opt.opt_op], feed_dict=feed_dict)

    avg_cost = outs[1]
    avg_accuracy = outs[0]

    if FLAGS.verbose and (epoch + 1) % 50 == 0:
        print("Epoch:", '%04d' % (epoch + 1), "train_loss=", "{:.5f}".format(avg_cost),
              "train_acc=", "{:.5f}".format(avg_accuracy))

def plot_graph(A):
    G = nx.from_numpy_matrix(A)
    indices = np.ndindex(7, 35)
    pos = dict(zip(G, indices))
    nx.draw(G, pos, node_size = 50)
    plt.show()
    plt.close()

gen = sess.run(model.sample(), feed_dict = feed_dict)
gen = np.ceil(gen - 0.9)
plot_graph(gen)


