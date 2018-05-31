from __future__ import division
from __future__ import print_function

import time
import os
import sys

import tensorflow as tf
import numpy as np
import scipy.sparse as sp
import scipy.stats as stats

from sklearn.metrics import roc_auc_score
from sklearn.metrics import average_precision_score
from sklearn.preprocessing import normalize

from sklearn import manifold
from scipy.special import expit

from optimizer import OptimizerVAE
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
flags.DEFINE_float('autoregressive_scalar', 0., 'Scalar for Graphite')
flags.DEFINE_integer('vae', 1, '1 for variational objective')

flags.DEFINE_integer('subsample', 0, 'Subsample in optimizer')
flags.DEFINE_float('subsample_frac', 1, 'Ratio of sampled non-edges to edges if using subsampling')

flags.DEFINE_integer('verbose', 1, 'Output all epoch data')
flags.DEFINE_integer('test_count', 10, 'batch of tests')

flags.DEFINE_string('dataset', 'cora', 'Dataset string.')
flags.DEFINE_string('model', 'vgae', 'Model string.')
flags.DEFINE_integer('features', 0, 'Whether to use features (1) or not (0).')
flags.DEFINE_integer('gpu', -1, 'Which gpu to use')
flags.DEFINE_integer('seeded', 1, 'Set numpy random seed')
flags.DEFINE_integer('connected_split', 1, 'use split with training set always connected')


if FLAGS.seeded:
    np.random.seed(1)

dataset_str = FLAGS.dataset
model_str = FLAGS.model

# Load data
adj, features = load_data(dataset_str)

adj_def = adj

# Store original adjacency matrix (without diagonal entries) for later
adj_orig = adj
adj_orig = adj_orig - sp.dia_matrix((adj_orig.diagonal()[np.newaxis, :], [0]), shape=adj_orig.shape)
adj_orig.eliminate_zeros()

if FLAGS.features == 0:
    features = sp.identity(features.shape[0])  # featureless

features = sparse_to_tuple(features.tocoo())
num_features = features[2][1]
features_nonzero = features[1].shape[0]

rocs = np.zeros(FLAGS.test_count)
aps = np.zeros(FLAGS.test_count)

os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
if FLAGS.gpu == -1:
    os.environ['CUDA_VISIBLE_DEVICES'] = ''
    sess = tf.Session()
else:
    os.environ['CUDA_VISIBLE_DEVICES'] = str(FLAGS.gpu)
    gpu_options = tf.GPUOptions(allow_growth=True)
    sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options, allow_soft_placement=True))

for test in range(FLAGS.test_count):
    func = get_test_edges
    if FLAGS.connected_split == 0:
        func = mask_test_edges
        
    adj_train, train_edges, val_edges, val_edges_false, test_edges, test_edges_false = func(adj_def)
    val_edges = tuple(zip(*val_edges))
    val_edges_false = tuple(zip(*val_edges_false))
    test_edges = tuple(zip(*test_edges))
    test_edges_false = tuple(zip(*test_edges_false))
    adj = adj_train

    adj_norm = preprocess_graph(adj)

    placeholders = {
        'features': tf.sparse_placeholder(tf.float32),
        'adj': tf.sparse_placeholder(tf.float32),
        'adj_orig': tf.sparse_placeholder(tf.float32),
        'dropout': tf.placeholder_with_default(0., shape=()),
    }

    num_nodes = adj.shape[0]

    model = None
    if model_str == 'relnet':
        model = GCNModelRelnet(placeholders, num_features, num_nodes, features_nonzero)
    elif model_str == 'feedback':
        model = GCNModelFeedback(placeholders, num_features, num_nodes, features_nonzero)
    else:
        model = GCNModelVAE(placeholders, num_features, num_nodes, features_nonzero)

    adj_label = sparse_to_tuple(adj_train + sp.eye(adj_train.shape[0]))

    pos_weight = float(adj.shape[0] * adj.shape[0] - adj.sum()) / adj.sum()
    norm = adj.shape[0] * adj.shape[0] / float((adj.shape[0] * adj.shape[0] - adj.sum()) * 2)

    with tf.name_scope('optimizer'):
        opt = OptimizerVAE(preds=model.reconstructions,
                           labels=tf.reshape(tf.sparse_tensor_to_dense(placeholders['adj_orig'], validate_indices=False), [-1]),
                           model=model, num_nodes=num_nodes,
                           pos_weight=pos_weight,
                           norm=norm)


    sess.run(tf.global_variables_initializer())

    def sigmoid(x):
        return 1 / (1 + np.exp(-x))

    def reconstruct():
        feed_dict = construct_feed_dict(adj_norm, adj_label, features, placeholders)
        feed_dict.update({placeholders['dropout']: 0.})

        emb, recon = sess.run([model.z_mean, model.reconstructions_noiseless], feed_dict=feed_dict)
        return (emb, np.reshape(recon, (num_nodes, num_nodes)))

    def get_roc_score(edges_pos, edges_neg):

        emb, adj_rec = reconstruct()

        preds = sigmoid(adj_rec[edges_pos])
        preds_neg = sigmoid(adj_rec[edges_neg])

        preds_all = np.hstack([preds, preds_neg])
        labels_all = np.hstack([np.ones(len(preds)), np.zeros(len(preds))])
        try:
            roc_score = roc_auc_score(labels_all, preds_all)
        except ValueError:
            roc_score = -1
        try:
            ap_score = average_precision_score(labels_all, preds_all)
        except ValueError:
            ap_score = -1

        return roc_score, ap_score, emb



    val_metrics = np.zeros(FLAGS.epochs)
    test_rocs = np.zeros(FLAGS.epochs)
    test_aps = np.zeros(FLAGS.epochs)
    test_embs = []

    # Train model
    for epoch in range(FLAGS.epochs):

        if FLAGS.edge_dropout > 0:
            adj_train_mini = edge_dropout(adj, FLAGS.edge_dropout)
            adj_norm_mini = preprocess_graph(adj_train_mini)
        else:
            adj_norm_mini = adj_norm

        feed_dict = construct_feed_dict(adj_norm_mini, adj_label, features, placeholders)
        feed_dict.update({placeholders['dropout']: FLAGS.dropout})
        outs = sess.run([opt.opt_op, opt.cost, opt.accuracy], feed_dict=feed_dict)

        avg_cost = outs[1]
        avg_accuracy = outs[2]

        roc_curr, ap_curr, _ = get_roc_score(val_edges, val_edges_false)
        val_metrics[epoch] = roc_curr + ap_curr
        roc_score, ap_score, emb = get_roc_score(test_edges, test_edges_false)
        test_rocs[epoch] = roc_score
        test_aps[epoch] = ap_score
        test_embs.append(emb)

        if FLAGS.verbose:
            print("Epoch:", '%04d' % (epoch + 1), "train_loss=", "{:.5f}".format(avg_cost),
                  "train_acc=", "{:.5f}".format(avg_accuracy), "val_roc=", "{:.5f}".format(roc_curr),
                  "val_ap=", "{:.5f}".format(ap_curr),
                  "test_roc=", "{:.5f}".format(roc_score),
                  "test_ap=", "{:.5f}".format(ap_score))

    arg = np.argmax(val_metrics)
    rocs[test] = test_rocs[arg]
    aps[test] = test_aps[arg]

    if FLAGS.verbose or dataset_str == 'pubmed':
        print(arg)
        print(test_rocs[arg])
        print(test_aps[arg])
        sys.stdout.flush()
        if FLAGS.verbose:
            break
if not FLAGS.verbose:
    print((np.mean(rocs), stats.sem(rocs)))
    print((np.mean(aps), stats.sem(aps)))
