from layers import GraphConvolution, GraphConvolutionSparse, InnerProductDecoder
from layers import *
import tensorflow as tf

flags = tf.app.flags
FLAGS = flags.FLAGS


class Model(object):
    def __init__(self, **kwargs):
        allowed_kwargs = {'name', 'logging'}
        for kwarg in kwargs.keys():
            assert kwarg in allowed_kwargs, 'Invalid keyword argument: ' + kwarg

        for kwarg in kwargs.keys():
            assert kwarg in allowed_kwargs, 'Invalid keyword argument: ' + kwarg
        name = kwargs.get('name')
        if not name:
            name = self.__class__.__name__.lower()
        self.name = name

        logging = kwargs.get('logging', False)
        self.logging = logging

        self.vars = {}

    def _build(self):
        raise NotImplementedError

    def build(self):
        """ Wrapper for _build() """
        with tf.variable_scope(self.name):
            self._build()
        variables = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=self.name)
        self.vars = {var.name: var for var in variables}

    def fit(self):
        pass

    def predict(self):
        pass

class GCNModelVAE(Model):
    '''VGAE Model for reconstructing graph edges from node representations.'''
    def __init__(self, placeholders, num_features, num_nodes, features_nonzero, **kwargs):
        super(GCNModelVAE, self).__init__(**kwargs)

        self.inputs = placeholders['features']
        self.input_dim = num_features
        self.features_nonzero = features_nonzero
        self.n_samples = num_nodes
        self.adj = placeholders['adj']
        self.dropout = placeholders['dropout']
        self.adj_label = placeholders['adj_orig']
        self.weight_norm = 0
        self.build()

    def encoder(self, inputs):

        hidden1 = GraphConvolutionSparse(input_dim=self.input_dim,
                                              output_dim=FLAGS.hidden1,
                                              adj=self.adj,
                                              features_nonzero=self.features_nonzero,
                                              act=tf.nn.relu,
                                              dropout=0.,
                                              logging=self.logging)(inputs)

        self.z_mean = GraphConvolution(input_dim=FLAGS.hidden1,
                                       output_dim=FLAGS.hidden2,
                                       adj=self.adj,
                                       act=lambda x: x,
                                       dropout=self.dropout,
                                       logging=self.logging)(hidden1)

        self.z_log_std = GraphConvolution(input_dim=FLAGS.hidden1,
                                          output_dim=FLAGS.hidden2,
                                          adj=self.adj,
                                          act=lambda x: x,
                                          dropout=self.dropout,
                                          logging=self.logging)(hidden1)

    def get_z(self, random):

        z = self.z_mean + tf.random_normal([self.n_samples, FLAGS.hidden2]) * tf.exp(self.z_log_std)
        if not random or not FLAGS.vae:
          z = self.z_mean

        return z

    def make_decoder(self):
      return

    def decoder(self, z):

        reconstructions = InnerProductDecoder(input_dim=FLAGS.hidden2,
                                      act=lambda x: x,
                                      dropout=0.,
                                      logging=self.logging)(z)

        reconstructions = tf.reshape(reconstructions, [-1])
        return reconstructions

    def _build(self):
  
        self.encoder(self.inputs)
        self.make_decoder()
        z = self.get_z(random = True)
        z_noiseless = self.get_z(random = False)
        if not FLAGS.vae:
          z = z_noiseless

        self.reconstructions = self.decoder(z)
        self.reconstructions_noiseless = self.decoder(z_noiseless)

class GCNModelFeedback(GCNModelVAE):
    '''Graphite model for reconstructing graph edges from node representations and intermediate complete graph.'''
    def __init__(self, placeholders, num_features, num_nodes, features_nonzero, **kwargs):
        super(GCNModelFeedback, self).__init__(placeholders, num_features, num_nodes, features_nonzero, **kwargs)

    def make_decoder(self):
        self.l0 = GraphiteSparse(input_dim=self.input_dim,
                                      output_dim=FLAGS.hidden3,
                                      act=tf.nn.relu,
                                      dropout=0.,
                                      logging=self.logging)

        self.l1 = Graphite(input_dim=FLAGS.hidden2,
                                              output_dim=FLAGS.hidden3,
                                              act=tf.nn.relu,
                                              dropout=0.,
                                              logging=self.logging)

        self.l2 = Graphite(input_dim=FLAGS.hidden3,
                                              output_dim=FLAGS.hidden2,
                                              act=lambda x: x,
                                              dropout=self.dropout,
                                              logging=self.logging)

        self.l3 = InnerProductDecoder(input_dim=FLAGS.hidden2,
                                      act=lambda x: x,
                                      logging=self.logging)

        self.l4 = Scale(input_dim = FLAGS.hidden2, logging = self.logging)

    def decoder(self, z):

        # recon = self.l3(z)
        # recon = tf.nn.sigmoid(recon)

        # recon = self.l3(tf.nn.l2_normalize(z, dim = 1))
        # recon += tf.ones_like(recon)

        # d = tf.reduce_sum(recon, 1)
        # d = tf.pow(d, -0.5)
        # recon = tf.expand_dims(d, 0) * recon * tf.expand_dims(d, 1)

        recon_1 = tf.nn.l2_normalize(z, dim = 1)
        recon_2 = tf.ones_like(recon_1)
        recon_2 /= tf.sqrt(tf.reduce_sum(recon_2, axis = 1, keepdims = True))

        d = tf.matmul(recon_1, tf.expand_dims(tf.reduce_sum(recon_1, axis = 0), 1)) + tf.matmul(recon_2, tf.expand_dims(tf.reduce_sum(recon_2, axis = 0), 1))
        d = tf.pow(d, -0.5)
        recon_1 *= d
        recon_2 *= d

        update = self.l1((z, recon_1, recon_2)) + self.l0((self.inputs, recon_1, recon_2))
        update = self.l2((update, recon_1, recon_2))

        # update = tf.nn.l2_normalize(update, dim = 1)
        # update = z + FLAGS.autoregressive_scalar * update

        update = (1 - FLAGS.autoregressive_scalar) * z + FLAGS.autoregressive_scalar * update

        reconstructions = self.l3(update)
        reconstructions = tf.reshape(reconstructions, [-1])
        return reconstructions

    def sample(self):
        z = tf.random_normal([self.n_samples, FLAGS.hidden2])
        reconstruction = tf.nn.sigmoid(self.decoder(z))
        reconstruction = tf.reshape(reconstruction, [self.n_samples, self.n_samples])
        return reconstruction

class GCNModelSiemens(GCNModelVAE):
    '''Model for generating graphs from multiple training graphs'''
    def __init__(self, placeholders, num_features, num_nodes, features_nonzero, **kwargs):
        super(GCNModelSiemens, self).__init__(placeholders, num_features, num_nodes, features_nonzero, **kwargs)

    def make_decoder(self):

        self.l0 = Dense(input_dim=self.input_dim,
                              output_dim=FLAGS.hidden3,
                              act=tf.nn.elu,
                              dropout=0.,
                              bias=True,
                              logging=self.logging)

        self.l1 = Dense(input_dim=FLAGS.hidden2,
                                              output_dim=FLAGS.hidden3,
                                              act=tf.nn.elu,
                                              dropout=0.,
                                              bias=True,
                                              logging=self.logging)

        self.l2 = Dense(input_dim=FLAGS.hidden3,
                                              output_dim=FLAGS.hidden2,
                                              act=lambda x: x,
                                              dropout=self.dropout,
                                              bias=True,
                                              logging=self.logging)

        self.l3 = Dense(input_dim=2 * FLAGS.hidden2,
                                              output_dim=FLAGS.hidden3,
                                              act=tf.nn.elu,
                                              dropout=self.dropout,
                                              bias=True,
                                              logging=self.logging)

        self.l3p5 = Dense(input_dim=FLAGS.hidden3,
                                              output_dim=FLAGS.hidden3,
                                              act=tf.nn.elu,
                                              dropout=self.dropout,
                                              bias=True,
                                              logging=self.logging)

        self.l4 = Dense(input_dim=FLAGS.hidden3,
                                              output_dim=1,
                                              act=lambda x: x,
                                              dropout=self.dropout,
                                              bias=True,
                                              logging=self.logging)

        self.l5 = InnerProductDecoder(input_dim=FLAGS.hidden2,
                                      act=lambda x: x,
                                      logging=self.logging)

    def decoder(self, z):

        update = self.l1(z) + self.l0(tf.sparse_tensor_to_dense(self.inputs))
        update = self.l2(update)

        A = tf.abs(tf.expand_dims(update, 1) - tf.expand_dims(update, 0))
        B = tf.expand_dims(update, 1) + tf.expand_dims(update, 0)
        update = tf.concat((A,B), axis = 2)
        update = tf.reshape(update, [-1, 2 * FLAGS.hidden2])

        update = self.l3(update)
        update = self.l3p5(update)
        update = self.l4(update)

        reconstructions = tf.squeeze(update)
        self.full_recon = tf.nn.sigmoid(reconstructions)
        return reconstructions

    def sample(self):
        z = tf.random_normal([self.n_samples, FLAGS.hidden2])
        reconstruction = tf.nn.sigmoid(self.decoder(z))
        reconstruction = tf.reshape(reconstruction, [self.n_samples, self.n_samples])
        return reconstruction

