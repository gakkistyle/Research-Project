from tran import *
import tensorflow as tf

from tensorflow.python.ops.rnn_cell_impl import BasicLSTMCell
from tensorflow.contrib.legacy_seq2seq.python.ops.seq2seq import rnn_decoder
from tensorflow.python.ops.distributions.normal import Normal
from functions import *
import numpy as np
import torch.nn as nn
import math
import torch

def _weight_variable(shape):
    initial = tf.truncated_normal(shape=shape, stddev=0.01)
    return tf.Variable(initial)


def _bias_variable(shape):
    initial = tf.constant(0.0, shape=shape)
    return tf.Variable(initial)

def _log_likelihood(loc_means, locs, variance):
    loc_means = tf.stack(loc_means)  # [timesteps, batch_sz, loc_dim]
    locs = tf.stack(locs)
    gaussian = Normal(loc_means, variance)
    logll = gaussian._log_prob(x=locs)  # [timesteps, batch_sz, loc_dim]
    logll = tf.reduce_sum(logll, 2)
    return tf.transpose(logll)  # [batch_sz, timesteps]

class PureTrans(object):
    def __init__(self,  img_width, img_height, nb_classes, learning_rate, learning_rate_decay_factor,
                 min_learning_rate, nb_training_batch, max_gradient_norm, is_training=False):

        self.img_ph = tf.compat.v1.placeholder(tf.float32, [None, img_height, img_width])
        self.lbl_ph = tf.compat.v1.placeholder(tf.int64, [None])

        self.global_step = tf.Variable(0, trainable=False)
        self.d_model = 120

        batch_size = tf.shape(self.img_ph)[0]

        # decayed_learning_rate = learning_rate * decay_rate ^ (global_step / training_batch_num)
        self.learning_rate = tf.maximum(tf.train.exponential_decay(
            learning_rate, self.global_step,
            nb_training_batch, # batch number
            learning_rate_decay_factor,
            # If the argument staircase is True,
            # then global_step / decay_steps is an integer division
            # and the decayed learning rate follows a staircase function.
            staircase=True),
            min_learning_rate)

        
        with tf.variable_scope('linear_before_transfor'):
            logit_wb1 = _weight_variable((img_width, self.d_model//4))
            #logit_w = _weight_variable((self.d_model, nb_classes))
            logit_bb1 = _bias_variable((self.d_model//4,))
            logit_wb2 = _weight_variable((self.d_model//4,self.d_model))
            logit_bb2 = _bias_variable((self.d_model,))

        #print("img shape",tf.shape(self.img_ph))
        img_2d = tf.reshape(self.img_ph,[-1,img_width])
        #print("img_2d shape",tf.shape(img_2d))
        trans_input_2d_ = tf.nn.xw_plus_b(img_2d, logit_wb1, logit_bb1)
        trans_input_2d = tf.nn.xw_plus_b(trans_input_2d_, logit_wb2, logit_bb2)
        #print("trans_input_2d shape",tf.shape(trans_input_2d))
        trans_input = tf.reshape(trans_input_2d,[batch_size,img_height,self.d_model])
        #print("trans_input shape",tf.shape(trans_input))

       # temp_target = tf.random.uniform((img_height,batch_size, self.d_model), minval=-1, maxval=1) 
        with tf.variable_scope('transformer'):
            # trans_network = Transformer(
            #     num_layers=1, d_model=self.d_model, num_heads=1, dff=2048)
            encoder_net = Encoder(
                num_layers=3, d_model=self.d_model, num_heads=3, dff=2048)
        # tran_output, state = trans_network(trans_input, temp_target, training=is_training, 
        #                        enc_padding_mask=None, 
        #                        look_ahead_mask=None,
        #                        dec_padding_mask=None)
        tran_output = encoder_net(trans_input,training = is_training,mask = None)

        #batch_dim = 1
        #indices = [[0]*batch_size]
        tran_out_to_linear_ = tran_output[:,::img_height]
        tran_out_to_linear = tf.reshape(tran_out_to_linear_,[-1,self.d_model] )
        #tran_out_to_linear = tf.gather(params = tran_output,indices = indices,axis = 1)
        #print("tran_out_toli shape",tf.shape(tran_out_to_linear))
        
        with tf.variable_scope('linear_after_transfor'):
            logit_w1 = _weight_variable((self.d_model, self.d_model//4))
            #logit_w = _weight_variable((self.d_model, nb_classes))
            logit_b1 = _bias_variable((self.d_model//4,))
            logit_w2 = _weight_variable((self.d_model//4, nb_classes))
            logit_b2 = _weight_variable((nb_classes,))

        #output
        logits1 = tf.nn.xw_plus_b(tran_out_to_linear, logit_w1, logit_b1)
        logits2 = tf.nn.xw_plus_b(logits1, logit_w2, logit_b2)
      
        self.softmax = tf.nn.softmax(logits2)
        #print("softmax shape:",tf.shape(self.softmax))
        self.pred = tf.argmax(self.softmax, 1)
        #print("the pred is ",self.pred)
        self.accuracy = tf.reduce_mean(tf.cast(tf.equal(self.pred, self.lbl_ph), tf.float32))

        #输出为[batch,timestep,d_model] -> [batch,class]
        #baseline??

        if is_training:
            self.cross_entropy = tf.reduce_mean(
                tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.lbl_ph, logits=logits2))
            params = tf.trainable_variables()
            gradients = tf.gradients(self.cross_entropy, params)
            clipped_gradients, norm = tf.clip_by_global_norm(gradients, max_gradient_norm)
            self.train_op = tf.train.AdamOptimizer(self.learning_rate).apply_gradients(
                zip(clipped_gradients, params), global_step=self.global_step)

        self.saver = tf.train.Saver(tf.global_variables(), max_to_keep=99999999)

class PositionalEncoding(nn.Module):

    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(
            0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)

class PureTran_torch(nn.Module):
    def __init__(self, height,nb_features,ntoken, ninp, nhead, nhid, nlayers,pe = False,
                 dropout=0.5):
        super(PureTran_torch, self).__init__()
        from torch.nn import TransformerEncoder, TransformerEncoderLayer
        self.model_type = 'Transformer'
        self.src_mask = None
        self.height = height
        self.nb_features= nb_features  #for linear transformation
        self.encoder = nn.Sequential(nn.Linear(nb_features, ninp//4),
                                     nn.Linear(ninp//4, ninp))
        #self.pos_encoder = PositionalEncoding(ninp, dropout)
        self.has_pe = pe
        self.pos_encoder = PositionalEncoding(ninp, dropout)
        encoder_layers = TransformerEncoderLayer(ninp, nhead, nhid, dropout)
        self.transformer_encoder = TransformerEncoder(encoder_layers, nlayers)
        
        self.ninp = ninp
        self.decoder = nn.Sequential(nn.Linear(ninp, ninp//4),
                                     nn.Linear(ninp//4, ntoken))
        # self.mesh_grid = MeshGrid()

    def _generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float(
            '-inf')).masked_fill(mask == 1, float(0.0))
        return mask

    def init_weights(self):
        initrange = 0.1
        self.encoder.weight.data.uniform_(-initrange, initrange)
        self.decoder.bias.data.zero_()
        self.decoder.weight.data.uniform_(-initrange, initrange)

    def forward(self, src):
        # if self.src_mask is None or self.src_mask.size(0) != len(src):
        #     device = src.device
        #     mask = self._generate_square_subsequent_mask(len(src)).to(device)
        #     self.src_mask = mask

      #  src = self.encoder(src) * math.sqrt(self.ninp)
       # torch.reshape(src,(-1,))
       # print("src shape:",src.shape)
       # print("num of features:",self.nb_features)
        src = self.encoder(src)
        if self.has_pe:
          src = self.pos_encoder(src)
        #src.reshape()
        output = self.transformer_encoder(src, mask=self.src_mask)
        # output = self.mesh_grid(
        #     self.transformer_encoder(src, mask=self.src_mask))
        output = output[:,::self.height]
        output = self.decoder(output)
        return output

