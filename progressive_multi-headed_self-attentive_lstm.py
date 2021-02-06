################################################   Author - Saran Karthikeyan     #######################################################
################################################ Thanks to Google, tensorflow and Keras    ##############################################
################################################ Welcome for Collaborative work   #######################################################
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

import collections
import warnings

import numpy as np
from keras import activations
from keras import backend as K
from keras import constraints
from keras import initializers
from keras import regularizers
from keras.layers import RNN
from keras.layers.advanced_activations import softmax
from keras.engine.base_layer import Layer
from keras.engine.input_spec import InputSpec
from keras.utils import tf_utils
from tensorflow.python.keras.layers.recurrent import DropoutRNNCellMixin
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.training.tracking import data_structures
from tensorflow.python.util.tf_export import keras_export

"""
Classes for LSTMCells for each column in Progressive Neural Network Architecture. The cell state formulation for 
each extended LSTM cell varies depending on the previous column LSTMCell values with self attention

LSTMCell has different calculations of cell formulation for each column and number of layer. Hence three classses 
have defined for three separate column. 

PROG_ATTLSTMCELL - LSTMCell for initial column and it has cell state formulation as same as built-in LSTMCell of keras
PROG_ATTLSTMCELL_1 - LSTMCell for extended column 1, which has cell state formulation depending on built-in LSTMCell
                  and previous cell state of initial column according to Progressive Neural Network
PROG_ATTLSTMCELL_2 - LSTMcell for extended column 2, which has cell stae formulation depending on built-in LSTMCell of
                  keras, and previous cell state of initial and previous column according to Progressive Neural Network

References:
- [Long short-term memory](
    http://www.bioinf.jku.at/publications/older/2604.pdf)
- [Progressive Neural Networks](
    https://arxiv.org/pdf/1606.04671.pdf)
- [Supervised sequence labeling with recurrent neural networks](
    http://www.cs.toronto.edu/~graves/preprint.pdf)
- [A Theoretically Grounded Application of Dropout in
    Recurrent Neural Networks](https://arxiv.org/abs/1512.05287)
- [Efficient Estimation of Word Representations in Vector Space](
    https://arxiv.org/pdf/1301.3781.pdf)
- [Long Short-Term Memory Networks for Machine Reading](
    https://www.aclweb.org/anthology/D16-1053.pdf)
    
"""

RECURRENT_DROPOUT_WARNING_MSG = (
    'RNN `implementation=2` is not supported when `recurrent_dropout` is set. '
    'Using `implementation=1`.')

@keras_export(v1=['keras.layers.LSTMCell'])
class LSTMCell(DropoutRNNCellMixin, Layer):

  def __init__(self,
               units,
               activation='tanh',
               recurrent_activation='hard_sigmoid',
               use_bias=True,
               kernel_initializer='glorot_uniform',
               recurrent_initializer='orthogonal',
               bias_initializer='zeros',
               unit_forget_bias=True,
               kernel_regularizer=None,
               recurrent_regularizer=None,
               bias_regularizer=None,
               kernel_constraint=None,
               recurrent_constraint=None,
               bias_constraint=None,
               frozen_weights = None,
               column = 0,
               layer = 0,               
               dropout=0.,
               recurrent_dropout=0.,
               **kwargs):
    # By default use cached variable under v2 mode, see b/143699808.
    if tf.compat.v1.executing_eagerly_outside_functions():
      self._enable_caching_device = kwargs.pop('enable_caching_device', True)
    else:
      self._enable_caching_device = kwargs.pop('enable_caching_device', False)
    super(LSTMCell, self).__init__(**kwargs)
    self.units = units
    self.activation = activations.get(activation)
    self.recurrent_activation = activations.get(recurrent_activation)
    self.use_bias = use_bias

    self.kernel_initializer = initializers.get(kernel_initializer)
    self.recurrent_initializer = initializers.get(recurrent_initializer)
    self.bias_initializer = initializers.get(bias_initializer)
    self.unit_forget_bias = unit_forget_bias

    self.kernel_regularizer = regularizers.get(kernel_regularizer)
    self.recurrent_regularizer = regularizers.get(recurrent_regularizer)
    self.bias_regularizer = regularizers.get(bias_regularizer)

    self.kernel_constraint = constraints.get(kernel_constraint)
    self.recurrent_constraint = constraints.get(recurrent_constraint)
    self.bias_constraint = constraints.get(bias_constraint)
    self.layer = layer
    self.column = column
    self.frozen_weights  = None
    self.dropout = min(1., max(0., dropout))
    self.recurrent_dropout = min(1., max(0., recurrent_dropout))
    implementation = kwargs.pop('implementation', 1)
    if self.recurrent_dropout != 0 and implementation != 1:
      logging.debug(RECURRENT_DROPOUT_WARNING_MSG)
      self.implementation = 1
    else:
      self.implementation = implementation
    # tuple(_ListWrapper) was silently dropping list content in at least 2.7.10,
    # and fixed after 2.7.16. Converting the state_size to wrapper around
    # NoDependency(), so that the base_layer.__setattr__ will not convert it to
    # ListWrapper. Down the stream, self.states will be a list since it is
    # generated from nest.map_structure with list, and tuple(list) will work
    # properly.
    self.state_size = data_structures.NoDependency([self.units, self.units])
    self.output_size = self.units

  @tf_utils.shape_type_conversion
  def build(self, input_shape):
    default_caching_device = _caching_device(self)
    input_dim = input_shape[-1]
    self.Qkernel = self.add_weight(
        shape=(input_dim, self.units * 4),
        name='kernel',
        initializer=self.kernel_initializer,
        regularizer=self.kernel_regularizer,
        constraint=self.kernel_constraint,
        caching_device=default_caching_device)

    self.Vkernel = self.add_weight(
        shape=(input_dim, self.units * 4),
        name='kernel',
        initializer=self.kernel_initializer,
        regularizer=self.kernel_regularizer,
        constraint=self.kernel_constraint,
        caching_device=default_caching_device)
    
    self.hidden_recurrent_kernel = self.add_weight(
        shape=(self.units, self.units * 4),
        name='recurrent_kernel',
        initializer=self.recurrent_initializer,
        regularizer=self.recurrent_regularizer,
        constraint=self.recurrent_constraint,
        caching_device=default_caching_device)

    self.cell_recurrent_kernel = self.add_weight(
        shape=(self.units, self.units * 4),
        name='recurrent_kernel',
        initializer=self.recurrent_initializer,
        regularizer=self.recurrent_regularizer,
        constraint=self.recurrent_constraint,
        caching_device=default_caching_device)

    if self.use_bias:
      if self.unit_forget_bias:
	      def bias_initializer(_, *args, **kwargs):
          return K.concatenate([
              self.bias_initializer((self.units,), *args, **kwargs),
              initializers.get('ones')((self.units,), *args, **kwargs),
              self.bias_initializer((self.units * 2,), *args, **kwargs),
          ])
      else:
        bias_initializer = self.bias_initializer
      self.bias = self.add_weight(
          shape=(self.units * 4,),
          name='bias',
          initializer=bias_initializer,
          regularizer=self.bias_regularizer,
          constraint=self.bias_constraint,
          caching_device=default_caching_device)
    else:
      self.bias = None
    self.built = True

    if self.column > 1 and self.layer > 1:
        if self.frozen_weights is not None:
            if len(frozen_weights) == 1:
                if self.use_bias:
                    if self.unit_forget_bias:
                        for i, trained_weights in enumerate(self.forzen_weights):
                            self.trained_Qkernel1, self.trained_Vkernel1, self.trained_hrecurrent_kernel1, self.trained_crecurrent_kernel1, self.bias1 = self.trained_weights
                        self.k_qti1, self.k_qtf1, self.k_qtc1, self.k_qto1 = tf.split(self.trained_Qkernel1, num_or_size_splits=4, axis=1)
                        self.k_vti1, self.k_vtf1, self.k_vtc1, self.k_vto1 = tf.split(self.trained_Vkernel1, num_or_size_splits=4, axis=1)
                        self.b_ti1, self.b_tf1, self.b_tc1, self.b_to1 = tf.split(self.bias1, num_or_size_splits=4, axis=0)
                        self.h_rk_ti1 = self.trained_hrecurrent_kernel1[:, : self.units]
                        self.h_rk_tf1 = self.trained_hrecurrent_kernel1[:,self.units:self.units*2]
                        self.h_rk_tc1 = self.trained_hrecurrent_kernel1[:,self.units*2:self.units*3]
                        self.h_rk_to1 = self.trained_hrecurrent_kernel1[:,self.units*3:]
                        self.c_rk_ti1 = self.trained_crecurrent_kernel1[:, : self.units]
                        self.c_rk_tf1 = self.trained_crecurrent_kernel1[:,self.units:self.units*2]
                        self.c_rk_tc1 = self.trained_crecurrent_kernel1[:,self.units*2:self.units*3]
                        self.c_rk_to1 = self.trained_crecurrent_kernel1[:,self.units*3:]
                else:
                    self.trained_Qkernel1,self.trained_Kkernel1,self.trained_Vkernel1, self.trained_recurrent_kernel1 = trained_weights
                    self.bias1 = None
                    self.k_qti1, self.k_qtf1, self.k_qtc1, self.k_qto1 = tf.split(self.trained_Qkernel1, num_or_size_splits=4, axis=1)
                    self.k_vti1, self.k_vtf1, self.k_vtc1, self.k_vto1 = tf.split(self.trained_Vkernel1, num_or_size_splits=4, axis=1)
                    self.b_ti1 = None
                    self.b_tf1 = None 
                    self.b_tc1 = None 
                    self.b_to1 = None
                    self.h_rk_ti1 = self.trained_hrecurrent_kernel1[:, : self.units]
                    self.h_rk_tf1 = self.trained_hrecurrent_kernel1[:,self.units:self.units*2]
                    self.h_rk_tc1 = self.trained_hrecurrent_kernel1[:,self.units*2:self.units*3]
                    self.h_rk_to1 = self.trained_hrecurrent_kernel1[:,self.units*3:]
                    self.c_rk_ti1 = self.trained_crecurrent_kernel1[:, : self.units]
                    self.c_rk_tf1 = self.trained_crecurrent_kernel1[:,self.units:self.units*2]
                    self.c_rk_tc1 = self.trained_crecurrent_kernel1[:,self.units*2:self.units*3]
                    self.c_rk_to1 = self.trained_crecurrent_kernel1[:,self.units*3:]
            else:
                if self.use_bias:
                    if self.unit_forget_bias:
                        for i, trained_weights in enumerate(self.frozen_weights):
                            if i == 1:
                                self.trained_Qkernel1, self.trained_Vkernel1, self.trained_hrecurrent_kernel1, self.trained_crecurrent_kernel1, self.bias1 = self.trained_weights
                            if i == 2:
                                self.trained_Qkernel2, self.trained_Vkernel2, self.trained_hrecurrent_kernel2, self.trained_crecurrent_kernel1, self.bias2 = self.trained_weights

                        self.k_qti1, self.k_qtf1, self.k_qtc1, self.k_qto1 = tf.split(self.trained_Qkernel1, num_or_size_splits=4, axis=1)
                        self.k_vti1, self.k_vtf1, self.k_vtc1, self.k_vto1 = tf.split(self.trained_Vkernel1, num_or_size_splits=4, axis=1)
                        self.b_ti1, self.b_tf1, self.b_tc1, self.b_to1 = tf.split(self.bias1, num_or_size_splits=4, axis=0)
                        self.h_rk_ti1 = self.trained_hrecurrent_kernel1[:, : self.units]
                        self.h_rk_tf1 = self.trained_hrecurrent_kernel1[:,self.units:self.units*2]
                        self.h_rk_tc1 = self.trained_hrecurrent_kernel1[:,self.units*2:self.units*3]
                        self.h_rk_to1 = self.trained_hrecurrent_kernel1[:,self.units*3:]
                        self.c_rk_ti1 = self.trained_crecurrent_kernel1[:, : self.units]
                        self.c_rk_tf1 = self.trained_crecurrent_kernel1[:,self.units:self.units*2]
                        self.c_rk_tc1 = self.trained_crecurrent_kernel1[:,self.units*2:self.units*3]
                        self.c_rk_to1 = self.trained_crecurrent_kernel1[:,self.units*3:]

                        self.k_qti2, self.k_qtf2, self.k_qtc2, self.k_qto2 = tf.split(self.trained_Qkernel2, num_or_size_splits=4, axis=1)
                        self.k_vti2, self.k_vtf2, self.k_vtc2, self.k_vto2 = tf.split(self.trained_Vkernel2, num_or_size_splits=4, axis=1)
                        self.b_ti2, self.b_tf2, self.b_tc2, self.b_to2 = tf.split(self.bias2, num_or_size_splits=4, axis=0)
                        self.h_rk_ti2 = self.trained_hrecurrent_kernel2[:, : self.units]
                        self.h_rk_tf2 = self.trained_hrecurrent_kernel2[:,self.units:self.units*2]
                        self.h_rk_tc2 = self.trained_hrecurrent_kernel2[:,self.units*2:self.units*3]
                        self.h_rk_to2 = self.trained_hrecurrent_kernel2[:,self.units*3:]
                        self.c_rk_ti2 = self.trained_crecurrent_kernel2[:, : self.units]
                        self.c_rk_tf2 = self.trained_crecurrent_kernel2[:,self.units:self.units*2]
                        self.c_rk_tc2 = self.trained_crecurrent_kernel2[:,self.units*2:self.units*3]
                        self.c_rk_to2 = self.trained_crecurrent_kernel2[:,self.units*3:]
                else:
                    for i, trained_weights in enumerate(self.forzen_weights):
                            if i == 1:
                                self.trained_Qkernel1, self.trained_Kkernel1, self.trained_Vkernel1, self.trained_recurrent_kernel1, self.bias1 = trained_weights
                                self.bias1 = None
                            if i == 2:
                                self.trained_Qkernel2, self.trained_Kkernel2, self.trained_Vkernel2, self.trained_recurrent_kernel2, self.bias2 = trained_weights
                                self.bias2 = None
                    
                    self.k_qti1, self.k_qtf1, self.k_qtc1, self.k_qto1 = tf.split(self.trained_Qkernel1, num_or_size_splits=4, axis=1)
                    self.k_vti1, self.k_vtf1, self.k_vtc1, self.k_vto1 = tf.split(self.trained_Vkernel1, num_or_size_splits=4, axis=1)
                    
                    self.h_rk_ti1 = self.trained_hrecurrent_kernel1[:, : self.units]
                    self.h_rk_tf1 = self.trained_hrecurrent_kernel1[:,self.units:self.units*2]
                    self.h_rk_tc1 = self.trained_hrecurrent_kernel1[:,self.units*2:self.units*3]
                    self.h_rk_to1 = self.trained_hrecurrent_kernel1[:,self.units*3:]
                    self.c_rk_ti1 = self.trained_crecurrent_kernel1[:, : self.units]
                    self.c_rk_tf1 = self.trained_crecurrent_kernel1[:,self.units:self.units*2]
                    self.c_rk_tc1 = self.trained_crecurrent_kernel1[:,self.units*2:self.units*3]
                    self.c_rk_to1 = self.trained_crecurrent_kernel1[:,self.units*3:]

                    self.k_qti2, self.k_qtf2, self.k_qtc2, self.k_qto2 = tf.split(self.trained_Qkernel2, num_or_size_splits=4, axis=1)
                    self.k_vti2, self.k_vtf2, self.k_vtc2, self.k_vto2 = tf.split(self.trained_Vkernel2, num_or_size_splits=4, axis=1)
                    
                    self.h_rk_ti2 = self.trained_hrecurrent_kernel2[:, : self.units]
                    self.h_rk_tf2 = self.trained_hrecurrent_kernel2[:,self.units:self.units*2]
                    self.h_rk_tc2 = self.trained_hrecurrent_kernel2[:,self.units*2:self.units*3]
                    self.h_rk_to2 = self.trained_hrecurrent_kernel2[:,self.units*3:]
                    self.c_rk_ti2 = self.trained_crecurrent_kernel2[:, : self.units]
                    self.c_rk_tf2 = self.trained_crecurrent_kernel2[:,self.units:self.units*2]
                    self.c_rk_tc2 = self.trained_crecurrent_kernel2[:,self.units*2:self.units*3]
                    self.c_rk_to2 = self.trained_crecurrent_kernel2[:,self.units*3:]
                    
                    self.b_ti1 = None
                    self.b_tf1 = None
                    self.b_tc1 = None
                    self.b_to1 = None
                    self.b_ti2 = None
                    self.b_tf2 = None
                    self.b_tc2 = None
                    self.b_to2 = None
    else:
      if self.use_bias:
        self.k_qti, self.k_qtf, self.k_qtc, self.k_qto = tf.split(self.trained_Qkernel, num_or_size_splits=4, axis=1)
        self.k_vti, self.k_vtf, self.k_vtc, self.k_vto = tf.split(self.trained_Vkernel, num_or_size_splits=4, axis=1)
        self.b_ti, self.b_tf, self.b_tc, self.b_to = tf.split(self.bias1, num_or_size_splits=4, axis=0)
        self.h_rk_ti = self.trained_hrecurrent_kernel[:, : self.units]
        self.h_rk_tf = self.trained_hrecurrent_kernel[:,self.units:self.units*2]
        self.h_rk_tc = self.trained_hrecurrent_kernel[:,self.units*2:self.units*3]
        self.h_rk_to = self.trained_hrecurrent_kernel[:,self.units*3:]
        self.c_rk_ti = self.trained_crecurrent_kernel[:, : self.units]
        self.c_rk_tf = self.trained_crecurrent_kernel[:,self.units:self.units*2]
        self.c_rk_tc = self.trained_crecurrent_kernel[:,self.units*2:self.units*3]
        self.c_rk_to = self.trained_crecurrent_kernel[:,self.units*3:]
      else:
        self.trained_Qkernel,self.trained_Kkernel,self.trained_Vkernel, self.trained_recurrent_kernel = self.trained_weights
        self.bias1 = None
        self.k_qti, self.k_qtf, self.k_qtc, self.k_qto = tf.split(self.trained_Qkernel, num_or_size_splits=4, axis=1)
        self.k_vti, self.k_vtf, self.k_vtc, self.k_vto = tf.split(self.trained_Vkernel, num_or_size_splits=4, axis=1)
        self.b_ti = None
        self.b_tf = None 
        self.b_tc = None 
        self.b_to = None
        self.h_rk_ti = self.trained_hrecurrent_kernel[:, : self.units]
        self.h_rk_tf = self.trained_hrecurrent_kernel[:,self.units:self.units*2]
        self.h_rk_tc = self.trained_hrecurrent_kernel[:,self.units*2:self.units*3]
        self.h_rk_to = self.trained_hrecurrent_kernel[:,self.units*3:]
        self.c_rk_ti = self.trained_crecurrent_kernel[:, : self.units]
        self.c_rk_tf = self.trained_crecurrent_kernel[:,self.units:self.units*2]
        self.c_rk_tc = self.trained_crecurrent_kernel[:,self.units*2:self.units*3]
        self.c_rk_to = self.trained_crecurrent_kernel[:,self.units*3:]

  def _hidden_cell_state(self, query, value, states, training = None):
    h_tm1 = states[0]  # previous memory state
    c_tm1 = states[1]  # previous carry state

    dp_mask = self.get_dropout_mask_for_cell(query, training, count=4)
    dp_mask_v = self.get_dropout_mask_for_cell(value, training, count=4)
    rec_dp_mask = self.get_recurrent_dropout_mask_for_cell(
        h_tm1, training, count=4)
    cell_dp_mask = self.get_recurrent_dropout_mask_for_cell(
        c_tm1, training, count=4)
    
    hs = (h_tm1, c_tm1, dp_mask, dp_mask_v, rec_dp_mask, cell_dp_mask)
    return hs
  
  def queryGateMask (self, query, dp_mask):
    if self.implementation == 1:
      if 0 < self.dropout < 1.:
        q_i = query * dp_mask[0]
        q_f = query * dp_mask[1]
        q_c = query * dp_mask[2]
        q_o = query * dp_mask[3]
      else:
        q_i = query
        q_f = query 
        q_c = query 
        q_o = query 
    q = (q_i, q_f, q_c, q_o)
    return q

  def valueGateMask (self, value, dp_mask_v):
    if self.implementation == 1:
      if 0 < self.dropout < 1.:
        v_i = value * dp_mask_v[0]
        v_f = value * dp_mask_v[1]
        v_c = value * dp_mask_v[2]
        v_o = value * dp_mask_v[3]
      else:
        v_i = value
        v_f = value 
        v_c = value 
        v_o = value
    v_i = K.transpose(v_i)
    v_f = K.transpose(v_f)
    v_c = K.transpose(v_c)
    v_o = K.transpose(v_o)
    v = (v_i, v_f, v_c, v_o)
    return v
  
  def _hidden_gateMask (self, query, value, states, training = None):
    h_tm1, c_tm1, dp_mask, dp_mask_v, rec_dp_mask, cell_dp_mask = self._hidden_cell_state(query, value, states, training = None)
    if 0 < self.recurrent_dropout < 1.:
      h_tm1_i = h_tm1 * rec_dp_mask[0]
      h_tm1_f = h_tm1 * rec_dp_mask[1]
      h_tm1_c = h_tm1 * rec_dp_mask[2]
      h_tm1_o = h_tm1 * rec_dp_mask[3]
    else:
      h_tm1_i = h_tm1
      h_tm1_f = h_tm1
      h_tm1_c = h_tm1
      h_tm1_o = h_tm1
    ph_tm1 = (h_tm1_i, h_tm1_f, h_tm1_c, h_tm1_o)
    return ph_tm1
  
  def _cell_gateMask (self, query, value, states, training = None):
    h_tm1, c_tm1, dp_mask, dp_mask_v, rec_dp_mask, cell_dp_mask = self._hidden_cell_state(query, value, states, training = None)
    if 0 < self.recurrent_dropout < 1.:
      c_tm1_i = h_tm1 * cell_dp_mask[0]
      c_tm1_f = h_tm1 * cell_dp_mask[1]
      c_tm1_c = h_tm1 * cell_dp_mask[2]
      c_tm1_o = h_tm1 * cell_dp_mask[3]
    else:
      c_tm1_i = h_tm1
      c_tm1_f = h_tm1
      c_tm1_c = h_tm1
      c_tm1_o = h_tm1
    c_tm1 = (c_tm1_i, c_tm1_f, c_tm1_c, c_tm1_o)
    return pc_tm1
  
  def weighted_avg_hiddenState(self):
    wa_hs_i = K.dot(self.h_rk_ti,self.k_vti)/K.sum(self.h_rk_ti,self.k_vti)
    wa_hs_f = K.dot(self.h_rk_tf,self.k_vtf)/K.sum(self.h_rk_tf,self.k_vtf)
    wa_hs_c = K.dot(self.h_rk_tc,self.k_vtc)/K.sum(self.h_rk_tc,self.k_vtc)
    wa_hs_o = K.dot(self.h_rk_to,self.k_vto)/K.sum(self.h_rk_to,self.k_vto)
    wa_hs = (wa_hs_i, wa_hs_f, wa_hs_c, wa_hs_o)

    return wa_hs
  
  def weighted_avg_hiddenState1(self):
    wa_hs_i1 = K.dot(self.h_rk_ti1,self.k_vti1)/K.sum(self.h_rk_ti1,self.k_vti1)
    wa_hs_f1 = K.dot(self.h_rk_tf1,self.k_vtf1)/K.sum(self.h_rk_tf1,self.k_vtf1)
    wa_hs_c1 = K.dot(self.h_rk_tc1,self.k_vtc1)/K.sum(self.h_rk_tc1,self.k_vtc1)
    wa_hs_o1 = K.dot(self.h_rk_to1,self.k_vto1)/K.sum(self.h_rk_to1,self.k_vto1)
    wa_hs1 = (wa_hs_i1, wa_hs_f1, wa_hs_c1, wa_hs_o1)

    return wa_hs1
  
  def weighted_avg_hiddenState2(self):
    wa_hs_i2 = K.dot(self.h_rk_ti2,self.k_vti2)/K.sum(self.h_rk_ti2,self.k_vti2)
    wa_hs_f2 = K.dot(self.h_rk_tf2,self.k_vtf2)/K.sum(self.h_rk_tf2,self.k_vtf2)
    wa_hs_c2 = K.dot(self.h_rk_tc2,self.k_vtc2)/K.sum(self.h_rk_tc2,self.k_vtc2)
    wa_hs_o2 = K.dot(self.h_rk_to2,self.k_vto2)/K.sum(self.h_rk_to2,self.k_vto2)
    wa_hs2 = (wa_hs_i2, wa_hs_f2, wa_hs_c2, wa_hs_o2)

    return wa_hs2

  def weighted_avg_cellState(self):
    wa_cs_i = K.dot(self.c_rk_ti,self.k_vti)/K.sum(self.c_rk_ti,self.k_vti)
    wa_cs_f = K.dot(self.c_rk_tf,self.k_vtf)/K.sum(self.c_rk_tf,self.k_vtf)
    wa_cs_c = K.dot(self.c_rk_tc,self.k_vtc)/K.sum(self.c_rk_tc,self.k_vtc)
    wa_cs_o = K.dot(self.c_rk_to,self.k_vto)/K.sum(self.c_rk_to,self.k_vto)
    wa_cs = (wa_cs_i, wa_cs_f, wa_cs_c, wa_cs_o)

    return wa_hs
  
  def weighted_avg_cellState1(self):
    wa_cs_i1 = K.dot(self.c_rk_ti1,self.k_vti1)/K.sum(self.c_rk_ti1,self.k_vti1)
    wa_cs_f1 = K.dot(self.c_rk_tf1,self.k_vtf1)/K.sum(self.c_rk_tf1,self.k_vtf1)
    wa_cs_c1 = K.dot(self.c_rk_tc1,self.k_vtc1)/K.sum(self.c_rk_tc1,self.k_vtc1)
    wa_cs_o1 = K.dot(self.c_rk_to1,self.k_vto1)/K.sum(self.c_rk_to1,self.k_vto1)
    wa_cs1 = (wa_cs_i1, wa_cs_f1, wa_cs_c1, wa_cs_o1)

    return wa_cs1
  
  def weighted_avg_cellState2(self):
    wa_cs_i2 = K.dot(self.c_rk_ti2,self.k_vti2)/K.sum(self.c_rk_ti2,self.k_vti2)
    wa_cs_f2 = K.dot(self.c_rk_tf2,self.k_vtf2)/K.sum(self.c_rk_tf2,self.k_vtf2)
    wa_cs_c2 = K.dot(self.c_rk_tc2,self.k_vtc2)/K.sum(self.c_rk_tc2,self.k_vtc2)
    wa_cs_o2 = K.dot(self.c_rk_to2,self.k_vto2)/K.sum(self.c_rk_to2,self.k_vto2)
    wa_cs2 = (wa_cs_i2, wa_cs_f2, wa_cs_c2, wa_cs_o2)

    return wa_cs2
  
  def weighted_avg_query(self):
    wa_q_i = K.dot(self.k_qti,self.k_vti)/K.sum(self.k_qti,self.k_vti)
    wa_q_f = K.dot(self.k_qtf,self.k_vtf)/K.sum(self.k_qtf,self.k_vtf)
    wa_q_c = K.dot(self.k_qtc,self.k_vtc)/K.sum(self.k_qtc,self.k_vtc)
    wa_q_o = K.dot(self.k_qto,self.k_vto)/K.sum(self.k_qto,self.k_vto)
    wa_q = (wa_q_i, wa_q_f, wa_q_c, wa_q_o)

    return wa_q
  
  def weighted_avg_query1(self):
    wa_q_i1 = K.dot(self.k_qti1,self.k_vti1)/K.sum(self.k_qti1,self.k_vti1)
    wa_q_f1 = K.dot(self.k_qtf1,self.k_vtf1)/K.sum(self.k_qtf1,self.k_vtf1)
    wa_q_c1 = K.dot(self.k_qtc1,self.k_vtc1)/K.sum(self.k_qtc1,self.k_vtc1)
    wa_q_o1 = K.dot(self.k_qto1,self.k_vto1)/K.sum(self.k_qto1,self.k_vto1)
    wa_q1 = (wa_q_i1, wa_q_f1, wa_q_c1, wa_q_o1)

    return wa_q1
  
  def weighted_avg_query2(self):
    wa_q_i2 = K.dot(self.k_qti2,self.k_vti2)/K.sum(self.k_qti2,self.k_vti2)
    wa_q_f2 = K.dot(self.k_qtf2,self.k_vtf2)/K.sum(self.k_qtf2,self.k_vtf2)
    wa_q_c2 = K.dot(self.k_qtc2,self.k_vtc2)/K.sum(self.k_qtc2,self.k_vtc2)
    wa_q_o2 = K.dot(self.k_qto2,self.k_vto2)/K.sum(self.k_qto2,self.k_vto2)
    wa_q2 = (wa_q_i2, wa_q_f2, wa_q_c2, wa_q_o2)

    return wa_q2

  def gateMechanism_hiddenState(self, query, value, states, Training = None):
    h_tm1, c_tm1, dp_mask, dp_mask_v, rec_dp_mask, cell_dp_mask = self._hidden_cell_state(query, value, states, training = None)
    wa_hs_i, wa_hs_f, wa_hs_c, wa_hs_o = self.weighted_average_hiddenState()
    v_i, v_f,v_c, v_o = self.valueGateMask(value, dp_mask_v)
    h_tm1_i, h_tm1_f, h_tm1_c, h_tm1_o = self._hidden_gateMask(query, value, states, training = None)

    gate_hs_i = K.dot(wa_hs_i,K.dot(h_tm1_i,v_i))
    gate_hs_f = K.dot(wa_hs_f,K.dot(h_tm1_f,v_f))
    gate_hs_c = K.dot(wa_hs_c,K.dot(h_tm1_c,v_c))
    gate_hs_o = K.dot(wa_hs_o,K.dot(h_tm1_o,v_o))

    gate_hs = (gate_hs_i, gate_hs_f, gate_hs_c, gate_hs_o)
    return gate_hs

  def gateMechanism_hiddenState1(self, query, value, states, Training = None):
    h_tm1, c_tm1, dp_mask, dp_mask_v, rec_dp_mask, cell_dp_mask = self._hidden_cell_state(query, value, states, training = None)
    wa_hs_i1, wa_hs_f1, wa_hs_c1, wa_hs_o1 = self.weighted_average_hiddenState1()
    v_i1, v_f1,v_c1, v_o1 = self.valueGateMask(value, dp_mask_v)
    h_tm1_i1, h_tm1_f1, h_tm1_c1, h_tm1_o1 = self._hidden_gateMask(query, value, states, training = None)

    gate_hs_i1 = K.dot(wa_hs_i1,K.dot(h_tm1_i1,v_i1))
    gate_hs_f1 = K.dot(wa_hs_f1,K.dot(h_tm1_f1,v_f1))
    gate_hs_c1 = K.dot(wa_hs_c1,K.dot(h_tm1_c1,v_c1))
    gate_hs_o1 = K.dot(wa_hs_o1,K.dot(h_tm1_o1,v_o1))

    gate_hs1 = (gate_hs_i1, gate_hs_f1, gate_hs_c1, gate_hs_o1)
    return gate_hs1

  def gateMechanism_hiddenState2(self, query, value, states, Training = None):
    h_tm1, c_tm1, dp_mask, dp_mask_v, rec_dp_mask, cell_dp_mask = self._hidden_cell_state(query, value, states, training = None)
    wa_hs_i2, wa_hs_f2, wa_hs_c2, wa_hs_o2 = self.weighted_average_hiddenState2()
    v_i2, v_f2,v_c2, v_o2 = self.valueGateMask(value, dp_mask_v)
    h_tm1_i2, h_tm1_f2, h_tm1_c2, h_tm1_o2 = self._hidden_gateMask(query, value, states, training = None)

    gate_hs_i2 = K.dot(wa_hs_i2,K.dot(h_tm1_i2,v_i2))
    gate_hs_f2 = K.dot(wa_hs_f2,K.dot(h_tm1_f2,v_f2))
    gate_hs_c2 = K.dot(wa_hs_c2,K.dot(h_tm1_c2,v_c2))
    gate_hs_o2 = K.dot(wa_hs_o2,K.dot(h_tm1_o2,v_o2))

    gate_hs2 = (gate_hs_i2, gate_hs_f2, gate_hs_c2, gate_hs_o2)
    return gate_hs2

  def gateMechanism_hiddenState(self, query, value, states, Training = None):
    h_tm1, c_tm1, dp_mask, dp_mask_v, rec_dp_mask, cell_dp_mask = self._hidden_cell_state(query, value, states, training = None)
    wa_cs_i, wa_cs_f, wa_cs_c, wa_cs_o = self.weighted_average_cellState()
    v_i, v_f,v_c, v_o = self.valueGateMask(value, dp_mask_v)
    c_tm1_i, c_tm1_f, c_tm1_c, c_tm1_o = self._hidden_gateMask(query, value, states, training = None)

    gate_cs_i = K.dot(wa_cs_i,K.dot(c_tm1_i,v_i))
    gate_cs_f = K.dot(wa_cs_f,K.dot(c_tm1_f,v_f))
    gate_cs_c = K.dot(wa_cs_c,K.dot(c_tm1_c,v_c))
    gate_cs_o = K.dot(wa_cs_o,K.dot(c_tm1_o,v_o))

    gate_cs = (gate_cs_i, gate_cs_f, gate_cs_c, gate_cs_o)
    return gate_cs

  def gateMechanism_hiddenState1(self, query, value, states, Training = None):
    h_tm1, c_tm1, dp_mask, dp_mask_v, rec_dp_mask, cell_dp_mask = self._hidden_cell_state(query, value, states, training = None)
    wa_cs_i1, wa_cs_f1, wa_cs_c1, wa_cs_o1 = self.weighted_average_cellState1()
    v_i1, v_f1,v_c1, v_o1 = self.valueGateMask(value, dp_mask_v)
    c_tm1_i1, c_tm1_f1, c_tm1_c1, c_tm1_o1 = self._hidden_gateMask(query, value, states, training = None)

    gate_cs_i1 = K.dot(wa_cs_i1,K.dot(c_tm1_i1,v_i1))
    gate_cs_f1 = K.dot(wa_cs_f1,K.dot(c_tm1_f1,v_f1))
    gate_cs_c1 = K.dot(wa_cs_c1,K.dot(c_tm1_c1,v_c1))
    gate_cs_o1 = K.dot(wa_cs_o1,K.dot(c_tm1_o1,v_o1))

    gate_cs1 = (gate_cs_i1, gate_cs_f1, gate_cs_c1, gate_cs_o1)
    return gate_cs1

  def gateMechanism_cellState2(self, query, value, states, Training = None):
    h_tm1, c_tm1, dp_mask, dp_mask_v, rec_dp_mask, cell_dp_mask = self._hidden_cell_state(query, value, states, training = None)
    wa_cs_i2, wa_cs_f2, wa_cs_c2, wa_cs_o2 = self.weighted_average_cellState2()
    v_i2, v_f2,v_c2, v_o2 = self.valueGateMask(value, dp_mask_v)
    c_tm1_i2, c_tm1_f2, c_tm1_c2, c_tm1_o2 = self._cell_gateMask(query, value, states, training = None)

    gate_cs_i2 = K.dot(wa_cs_i2,K.dot(c_tm1_i2,v_i2))
    gate_cs_f2 = K.dot(wa_cs_f2,K.dot(c_tm1_f2,v_f2))
    gate_cs_c2 = K.dot(wa_cs_c2,K.dot(c_tm1_c2,v_c2))
    gate_cs_o2 = K.dot(wa_cs_o2,K.dot(c_tm1_o2,v_o2))

    gate_cs2 = (gate_cs_i2, gate_cs_f2, gate_cs_c2, gate_cs_o2)
    return gate_cs2
  
  def gateMechanism_query(self, query, value, states, Training = None):
    h_tm1, c_tm1, dp_mask, dp_mask_v, rec_dp_mask, cell_dp_mask = self._hidden_cell_state(query, value, states, training = None)
    wa_q_i, wa_q_f, wa_q_c, wa_q_o = self.weighted_average_query()
    v_i, v_f,v_c, v_o = self.valueGateMask(value, dp_mask_v)
    q_i, q_f, q_c, q_o = self.queryGateMask(query, dp_mask)

    gate_q_i = K.dot(wa_q_i,K.dot(q_i,v_i))
    gate_q_f = K.dot(wa_q_f,K.dot(q_f,v_f))
    gate_q_c = K.dot(wa_q_c,K.dot(q_c,v_c))
    gate_q_o = K.dot(wa_q_o,K.dot(q_o,v_o))

    if self.use_bias:
      gate_q_i = K.bias_add(gate_q_i, self.b_ti)
      gate_q_f = K.bias_add(gate_q_f, self.b_tf)
      gate_q_c = K.bias_add(gate_q_c, self.b_tc)
      gate_q_o = K.bias_add(gate_q_o, self.b_to)

    gate_q = (gate_q_i, gate_q_f, gate_q_c, gate_q_o)
    return gate_q

  def gateMechanism_query1(self, query, value, states, Training = None):
    h_tm1, c_tm1, dp_mask, dp_mask_v, rec_dp_mask, cell_dp_mask = self._hidden_cell_state(query, value, states, training = None)
    wa_q_i1, wa_q_f1, wa_q_c1, wa_q_o1 = self.weighted_average_query1()
    v_i1, v_f1,v_c1, v_o1 = self.valueGateMask(value, dp_mask_v)
    q_i1, q_f1, q_c1, q_o1 = self.queryGateMask(query, dp_mask)

    gate_q_i1 = K.dot(wa_q_i1,K.dot(q_i1,v_i1))
    gate_q_f1 = K.dot(wa_q_f1,K.dot(q_f1,v_f1))
    gate_q_c1 = K.dot(wa_q_c1,K.dot(q_c1,v_c1))
    gate_q_o1 = K.dot(wa_q_o1,K.dot(q_o1,v_o1))
    if self.use_bias:
      gate_q_i1 = K.bias_add(gate_q_i1, self.b_ti1)
      gate_q_f1 = K.bias_add(gate_q_f1, self.b_tf1)
      gate_q_c1 = K.bias_add(gate_q_c1, self.b_tc1)
      gate_q_o1 = K.bias_add(gate_q_o1, self.b_to1)

    gate_q1 = (gate_q_i1, gate_q_f1, gate_q_c1, gate_q_o1)
    return gate_q1
  
  def gateMechanism_query2(self, query, value, states, Training = None):
    h_tm1, c_tm1, dp_mask, dp_mask_v, rec_dp_mask, cell_dp_mask = self._hidden_cell_state(query, value, states, training = None)
    wa_q_i2, wa_q_f2, wa_q_c2, wa_q_o2 = self.weighted_average_query2()
    v_i2, v_f2, v_c2, v_o2 = self.valueGateMask(value, dp_mask_v)
    q_i2, q_f2, q_c2, q_o2 = self.queryGateMask(query, dp_mask)

    gate_q_i2 = K.dot(wa_q_i2,K.dot(q_i2,v_i2))
    gate_q_f2 = K.dot(wa_q_f2,K.dot(q_f2,v_f2))
    gate_q_c2 = K.dot(wa_q_c2,K.dot(q_c2,v_c2))
    gate_q_o2 = K.dot(wa_q_o2,K.dot(q_o2,v_o2))

    if self.use_bias:
      gate_q_i2 = K.bias_add(gate_q_i2, self.b_ti2)
      gate_q_f2 = K.bias_add(gate_q_f2, self.b_tf2)
      gate_q_c2 = K.bias_add(gate_q_c2, self.b_tc2)
      gate_q_o2 = K.bias_add(gate_q_o2, self.b_to2)

    gate_q2 = (gate_q_i2, gate_q_f2, gate_q_c2, gate_q_o2)
    return gate_q2

  def multiHead_selfAttention(self, query, value, states, Training = None):
    h_tm1_i, h_tm1_f, h_tm1_c, h_tm1_o = self._hidden_gateMask(query, value, states, Training = None)
    gate_hs_i, gate_hs_f, gate_hs_c, gate_hs_o = self.gateMechanism_hiddenState(query, value, states, Training = None)
    gate_cs_i, gate_cs_f, gate_cs_c, gate_cs_o = self.gateMechanism_cellState(query, value, states, Training = None)
    gate_q_i, gate_q_f, gate_q_c, gate_q_o = self.gateMechanism_query(query, value, states, Training = None)

    att_out_i = K.dot(K.transpose(self.k_qti), self.recurrent_activation(K.sum(gate_hs_i, gate_q_i, gate_cs_i)))
    att_out_f = K.dot(K.transpose(self.k_qtf), self.recurrent_activation(K.sum(gate_hs_c, gate_q_f, gate_cs_f)))
    att_out_c = K.dot(K.transpose(self.k_qtc), self.recurrent_activation(K.sum(gate_hs_f, gate_q_c, gate_cs_c)))
    att_out_o = K.dot(K.transpose(self.k_qto), self.recurrent_activation(K.sum(gate_hs_o, gate_q_o, gate_cs_o)))

    mhsa_i = K.dot(self.h_rk_ti, K.dot(h_tm1_i,softmax(att_out_i)))
    mhsa_f = K.dot(self.h_rk_tf, K.dot(h_tm1_f,softmax(att_out_f)))
    mhsa_c = K.dot(self.h_rk_tc, K.dot(h_tm1_c,softmax(att_out_c)))
    mhsa_o = K.dot(self.h_rk_to, K.dot(h_tm1_o,softmax(att_out_o)))
    mhsa = (mhsa_i, mhsa_f, mhsa_c, mhsa_o)

    return mhsa
  
  def multiHead_selfAttention1(self, query, value, states, Training = None):
    h_tm1_i1, h_tm1_f1, h_tm1_c1, h_tm1_o1 = self._hidden_gateMask(query, value, states, Training = None)
    gate_hs_i1, gate_hs_f1, gate_hs_c1, gate_hs_o1 = self.gateMechanism_hiddenState1(query, value, states, Training = None)
    gate_cs_i1, gate_cs_f1, gate_cs_c1, gate_cs_o1 = self.gateMechanism_cellState1(query, value, states, Training = None)
    gate_q_i1, gate_q_f1, gate_q_c1, gate_q_o1 = self.gateMechanism_query1(query, value, states, Training = None)

    att_out_i1 = K.dot(K.transpose(self.k_qti1), self.recurrent_activation(K.sum(gate_hs_i1, gate_q_i1, gate_cs_i1)))
    att_out_f1 = K.dot(K.transpose(self.k_qtf1), self.recurrent_activation(K.sum(gate_hs_c1, gate_q_f1, gate_cs_f1)))
    att_out_c1 = K.dot(K.transpose(self.k_qtc1), self.recurrent_activation(K.sum(gate_hs_f1, gate_q_c1, gate_cs_c1)))
    att_out_o1 = K.dot(K.transpose(self.k_qto1), self.recurrent_activation(K.sum(gate_hs_o1, gate_q_o1, gate_cs_o1)))

    mhsa_i1 = K.dot(self.h_rk_ti1, K.dot(h_tm1_i1,softmax(att_out_i1)))
    mhsa_f1 = K.dot(self.h_rk_tf1, K.dot(h_tm1_f1,softmax(att_out_f1)))
    mhsa_c1 = K.dot(self.h_rk_tc1, K.dot(h_tm1_c1,softmax(att_out_c1)))
    mhsa_o1 = K.dot(self.h_rk_to1, K.dot(h_tm1_o1,softmax(att_out_o1)))

    mhsa1 = (mhsa_i1, mhsa_f1, mhsa_c1, mhsa_o1)

    return mhsa1
  
  def multiHead_selfAttention2(self, query, value, states, Training = None):
    h_tm1_i2, h_tm1_f2, h_tm1_c2, h_tm1_o2 = self._hidden_gateMask(query, value, states, Training = None)
    gate_hs_i2, gate_hs_f2, gate_hs_c2, gate_hs_o2 = self.gateMechanism_hiddenState2(query, value, states, Training = None)
    gate_cs_i2, gate_cs_f2, gate_cs_c2, gate_cs_o2 = self.gateMechanism_cellState2(query, value, states, Training = None)
    gate_q_i2, gate_q_f2, gate_q_c2, gate_q_o2 = self.gateMechanism_query2(query, value, states, Training = None)

    att_out_i2 = K.dot(K.transpose(self.k_qti2), self.recurrent_activation(K.sum(gate_hs_i2, gate_q_i2, gate_cs_i2)))
    att_out_f2 = K.dot(K.transpose(self.k_qtf2), self.recurrent_activation(K.sum(gate_hs_c2, gate_q_f2, gate_cs_f2)))
    att_out_c2 = K.dot(K.transpose(self.k_qtc2), self.recurrent_activation(K.sum(gate_hs_f2, gate_q_c2, gate_cs_c2)))
    att_out_o2 = K.dot(K.transpose(self.k_qto2), self.recurrent_activation(K.sum(gate_hs_o2, gate_q_o2, gate_cs_o2)))

    mhsa_i2 = K.dot(self.h_rk_ti2, K.dot(h_tm1_i2,softmax(att_out_i2)))
    mhsa_f2 = K.dot(self.h_rk_tf2, K.dot(h_tm1_f2,softmax(att_out_f2)))
    mhsa_c2 = K.dot(self.h_rk_tc2, K.dot(h_tm1_c2,softmax(att_out_c2)))
    mhsa_o2 = K.dot(self.h_rk_to2, K.dot(h_tm1_o2,softmax(att_out_o2)))

    mhsa2 = (mhsa_i2, mhsa_f2, mhsa_c2, mhsa_o2)

    return mhsa2
    
  def _compute_carry_and_output(self, query, value, states, c_tm1, Training = None):
    """Computes carry and output using split kernels."""
    mhsa_i, mhsa_f, mhsa_c, mhsa_o = self.MultiHead_selfAttention(query, value, states, Training = None)
    i = self.recurrent_activation(mhsa_i)
    f = self.recurrent_activation(mhsa_f)
    c = f * c_tm1 + i * self.activation(mhsa_c)
    o = self.recurrent_activation(mhsa_o)
    
    if self.column == 2 and self.layer > 1:
      mhsa_i1, mhsa_f1, mhsa_c1, mhsa_o1 = self.MultiHead_selfAttention1(query, value, states, Training = None)
      i = i + self.recurrent_activation(mhsa_i1)
      f = f + self.recurrent_activation(mhsa_f1)
      c = c + (f * c_tm1 + i * self.recurrent_activation(mhsa_c1))
      o = o + self.recurrent_activation(mhsa_o1)

    if self.column == 3 and self.layer > 1:
      mhsa_i1, mhsa_f1, mhsa_c1, mhsa_o1 = self.MultiHead_selfAttention1(query, value, states, Training = None)
      mhsa_i2, mhsa_f2, mhsa_c2, mhsa_o2 = self.MultiHead_selfAttention2(query, value, states, Training = None)
      i = i + self.recurrent_activation(mhsa_i1) + self.recurrent_activation(mhsa_i2)
      f = f + self.recurrent_activation(mhsa_f1) + self.recurrent_activation(mhsa_i2)
      c = c + (f * c_tm1 + i *self.recurrent_activation(mhsa_c1)) + (f * c_tm1 + i * self.recurrent_activation(mhsa_i2))
      o = o + self.recurrent_activation(mhsa_o1) + self.recurrent_activation(mhsa_i2)

    return c, o

  def _compute_carry_and_output_fused(self, z, c_tm1):
    """Computes carry and output using fused kernels."""
    z0, z1, z2, z3 = z
    i = self.recurrent_activation(z0)
    f = self.recurrent_activation(z1)
    c = f * c_tm1 + i * self.activation(z2)
    o = self.recurrent_activation(z3)
    return c, o

  def call(self, query, value, states, training=None):
    h_tm1, c_tm1, dp_mask, dp_mask_v, rec_dp_mask, cell_dp_mask
    if self.implementation == 1:
      c, o = self._compute_carry_and_output(query, value, states, c_tm1, training=None)
    else:
      if 0. < self.dropout < 1.:
        inputs = query * dp_mask[0]
        values = value * dp_mask_v[0]
      wa_q = K.dot(self.Qkernel, self.Vkernel)/K.sum(self.Qkernel, self.Vkernel)
      wa_hs = K.dot(self.hidden_recurrent_kernel, self.Vkernel)/K.sum(self.hidden_recurrent_kernel, self.Vkernel)
      wa_cs = K.dot(self.cell_recurrent_kernel, self.Vkernel)/K.sum(self.cell_recurrent_kernel, self.Vkernel)
      z = K.dot(inputs, K.dot(values, wa_q))
      z += K.dot(h_tm1, K.dot(values, wa_hs))
      z += K.dot(c_tm1, K.dot(values, wa_cs))
      if self.use_bias:
        z = K.bias_add(z, self.bias)

      z = tf.split(z, num_or_size_splits=4, axis=1)
      c, o = self._compute_carry_and_output_fused(z, c_tm1)

    h = o * self.activation(c)
    return h, [h, c]

@keras_export(v1=['keras.layers.PROG_MHSALSTM'])
class PROG_MHSALSTM(RNN):
  """Long Short-Term Memory layer - Hochreiter 1997.
   Note that this cell is not optimized for performance on GPU. Please use
  `tf.compat.v1.keras.layers.CuDNNLSTM` for better performance on GPU.
  Args:
    units: Positive integer, dimensionality of the output space.
    activation: Activation function to use.
      Default: hyperbolic tangent (`tanh`).
      If you pass `None`, no activation is applied
      (ie. "linear" activation: `a(x) = x`).
    recurrent_activation: Activation function to use
      for the recurrent step.
      Default: hard sigmoid (`hard_sigmoid`).
      If you pass `None`, no activation is applied
      (ie. "linear" activation: `a(x) = x`).
    use_bias: Boolean, whether the layer uses a bias vector.
    kernel_initializer: Initializer for the `kernel` weights matrix,
      used for the linear transformation of the inputs..
    recurrent_initializer: Initializer for the `recurrent_kernel`
      weights matrix,
      used for the linear transformation of the recurrent state.
    bias_initializer: Initializer for the bias vector.
    unit_forget_bias: Boolean.
      If True, add 1 to the bias of the forget gate at initialization.
      Setting it to true will also force `bias_initializer="zeros"`.
      This is recommended in [Jozefowicz et al., 2015](
        http://www.jmlr.org/proceedings/papers/v37/jozefowicz15.pdf).
    kernel_regularizer: Regularizer function applied to
      the `kernel` weights matrix.
    recurrent_regularizer: Regularizer function applied to
      the `recurrent_kernel` weights matrix.
    bias_regularizer: Regularizer function applied to the bias vector.
    activity_regularizer: Regularizer function applied to
      the output of the layer (its "activation").
    kernel_constraint: Constraint function applied to
      the `kernel` weights matrix.
    recurrent_constraint: Constraint function applied to
      the `recurrent_kernel` weights matrix.
    bias_constraint: Constraint function applied to the bias vector.
    dropout: Float between 0 and 1.
      Fraction of the units to drop for
      the linear transformation of the inputs.
    recurrent_dropout: Float between 0 and 1.
      Fraction of the units to drop for
      the linear transformation of the recurrent state.
    return_sequences: Boolean. Whether to return the last output.
      in the output sequence, or the full sequence.
    return_state: Boolean. Whether to return the last state
      in addition to the output.
    go_backwards: Boolean (default False).
      If True, process the input sequence backwards and return the
      reversed sequence.
    stateful: Boolean (default False). If True, the last state
      for each sample at index i in a batch will be used as initial
      state for the sample of index i in the following batch.
    unroll: Boolean (default False).
      If True, the network will be unrolled,
      else a symbolic loop will be used.
      Unrolling can speed-up a RNN,
      although it tends to be more memory-intensive.
      Unrolling is only suitable for short sequences.
    time_major: The shape format of the `inputs` and `outputs` tensors.
      If True, the inputs and outputs will be in shape
      `(timesteps, batch, ...)`, whereas in the False case, it will be
      `(batch, timesteps, ...)`. Using `time_major = True` is a bit more
      efficient because it avoids transposes at the beginning and end of the
      RNN calculation. However, most TensorFlow data is batch-major, so by
      default this function accepts input and emits output in batch-major
      form.
    column: Int (default: None), Max = 3
            If '1' cell formulation happens in PROG_ATTLSTMCELL (No influenze of previous column)
            If '2' cell formulation happens in both PROG_ATTLSTMCELL and PROG_ATTLSTMCELL_1 (Influenze of initial column 
            for layer '2')
            If '3' cell formulation happens in PROG_ATTLSTMCELL, PROG_ATTLSTMCELL_1 and PROG_ATTLSTMCELL_2 (Influenze of
            initial, and previous column for layer '2')
        layer: Int(default: None), Max = 2
            If '1' cell formulation happens in PROG_ATTLSTMCELL (No influenze of previous column)
            If '2' cell formulation depends on column value (Influenze of previous column)
    # References
        - [Long short-term memory](
          http://www.bioinf.jku.at/publications/older/2604.pdf)
        - [Learning to forget: Continual prediction with LSTM](
          http://www.mitpressjournals.org/doi/pdf/10.1162/089976600300015015)
        - [Supervised sequence labeling with recurrent neural networks](
          http://www.cs.toronto.edu/~graves/preprint.pdf)
        - [A Theoretically Grounded Application of Dropout in
           Recurrent Neural Networks](https://arxiv.org/abs/1512.05287)
  Call arguments:
    inputs: A 3D tensor.
    mask: Binary tensor of shape `(samples, timesteps)` indicating whether
      a given timestep should be masked. An individual `True` entry indicates
      that the corresponding timestep should be utilized, while a `False`
      entry indicates that the corresponding timestep should be ignored.
    training: Python boolean indicating whether the layer should behave in
      training mode or in inference mode. This argument is passed to the cell
      when calling it. This is only relevant if `dropout` or
      `recurrent_dropout` is used.
    initial_state: List of initial state tensors to be passed to the first
      call of the cell.
  """

  def __init__(self,
               units,
               activation='tanh',
               recurrent_activation='hard_sigmoid',
               use_bias=True,
               kernel_initializer='glorot_uniform',
               recurrent_initializer='orthogonal',
               bias_initializer='zeros',
               unit_forget_bias=True,
               kernel_regularizer=None,
               recurrent_regularizer=None,
               bias_regularizer=None,
               activity_regularizer=None,
               kernel_constraint=None,
               recurrent_constraint=None,
               bias_constraint=None,
               dropout=0.,
               recurrent_dropout=0.,
               return_sequences=False,
               return_state=False,
               go_backwards=False,
               stateful=False,
               unroll=False,
               **kwargs):
    implementation = kwargs.pop('implementation', 1)
    if implementation == 0:
      logging.warning('`implementation=0` has been deprecated, '
                      'and now defaults to `implementation=1`.'
                      'Please update your layer call.')
    if 'enable_caching_device' in kwargs:
      cell_kwargs = {'enable_caching_device':
                     kwargs.pop('enable_caching_device')}
    else:
      cell_kwargs = {}
    cell = LSTMCell(
        units,
        activation=activation,
        recurrent_activation=recurrent_activation,
        use_bias=use_bias,
        kernel_initializer=kernel_initializer,
        recurrent_initializer=recurrent_initializer,
        unit_forget_bias=unit_forget_bias,
        bias_initializer=bias_initializer,
        kernel_regularizer=kernel_regularizer,
        recurrent_regularizer=recurrent_regularizer,
        bias_regularizer=bias_regularizer,
        kernel_constraint=kernel_constraint,
        recurrent_constraint=recurrent_constraint,
        bias_constraint=bias_constraint,
        dropout=dropout,
        recurrent_dropout=recurrent_dropout,
        implementation=implementation,
        column=column,
        layer=layer,
        trained_weights=None,
        attention = 0,
        dtype=kwargs.get('dtype'),
        trainable=kwargs.get('trainable', True),
        **cell_kwargs)
    super(PROG_MHSALSTM, self).__init__(
        cell,
        return_sequences=return_sequences,
        return_state=return_state,
        go_backwards=go_backwards,
        stateful=stateful,
        unroll=unroll,
        **kwargs)
    self.activity_regularizer = regularizers.get(activity_regularizer)
    self.input_spec = [InputSpec(ndim=3)]

  def call(self, inputs, mask=None, training=None, initial_state=None):
    return super(PROG_MHSALSTM, self).call(
        inputs, mask=mask, training=training, initial_state=initial_state)

  @property
  def units(self):
    return self.cell.units

  @property
  def activation(self):
    return self.cell.activation

  @property
  def recurrent_activation(self):
    return self.cell.recurrent_activation

  @property
  def use_bias(self):
    return self.cell.use_bias

  @property
  def kernel_initializer(self):
    return self.cell.kernel_initializer

  @property
  def recurrent_initializer(self):
    return self.cell.recurrent_initializer

  @property
  def bias_initializer(self):
    return self.cell.bias_initializer

  @property
  def unit_forget_bias(self):
    return self.cell.unit_forget_bias

  @property
  def kernel_regularizer(self):
    return self.cell.kernel_regularizer

  @property
  def recurrent_regularizer(self):
    return self.cell.recurrent_regularizer

  @property
  def bias_regularizer(self):
    return self.cell.bias_regularizer

  @property
  def kernel_constraint(self):
    return self.cell.kernel_constraint

  @property
  def recurrent_constraint(self):
    return self.cell.recurrent_constraint

  @property
  def bias_constraint(self):
    return self.cell.bias_constraint

  @property
  def dropout(self):
    return self.cell.dropout

  @property
  def recurrent_dropout(self):
    return self.cell.recurrent_dropout

  @property
  def implementation(self):
    return self.cell.implementation
  
  @property
    def column(self):
        return self.cell.column
    
  @property
  def layer(self):
      return self.cell.layer

  @property
  def trained_weights(self):
      return self.cell.trained_weights

  @property
  def attention(self):
      return self.cell.attention

  def get_config(self):
    config = {
        'units':
            self.units,
        'activation':
            activations.serialize(self.activation),
        'recurrent_activation':
            activations.serialize(self.recurrent_activation),
        'use_bias':
            self.use_bias,
        'kernel_initializer':
            initializers.serialize(self.kernel_initializer),
        'recurrent_initializer':
            initializers.serialize(self.recurrent_initializer),
        'bias_initializer':
            initializers.serialize(self.bias_initializer),
        'unit_forget_bias':
            self.unit_forget_bias,
        'kernel_regularizer':
            regularizers.serialize(self.kernel_regularizer),
        'recurrent_regularizer':
            regularizers.serialize(self.recurrent_regularizer),
        'bias_regularizer':
            regularizers.serialize(self.bias_regularizer),
        'activity_regularizer':
            regularizers.serialize(self.activity_regularizer),
        'kernel_constraint':
            constraints.serialize(self.kernel_constraint),
        'recurrent_constraint':
            constraints.serialize(self.recurrent_constraint),
        'bias_constraint':
            constraints.serialize(self.bias_constraint),
        'dropout':
            self.dropout,
        'recurrent_dropout':
            self.recurrent_dropout,
        'implementation':
            self.implementation,
        'column': self.column,
        'layer': self.layer,
        'trained_weights':self.trained_weights,
        'attention':self.attention
    }
    config.update(_config_for_enable_caching_device(self.cell))
    base_config = super(LSTM, self).get_config()
    del base_config['cell']
    return dict(list(base_config.items()) + list(config.items()))

  @classmethod
  def from_config(cls, config):
    if 'implementation' in config and config['implementation'] == 0:
      config['implementation'] = 1
    return cls(**config)

    


     

