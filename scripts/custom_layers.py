#!/usr/bin/env python
# coding: utf-8

# # Invertible Convolution and WaveNet Custom Layers 

# ## Boilerplate
# Start with standard imports as well as adding the scripts directory to the system path to allow custom imports.

# In[3]:


import tensorflow as tf
from tensorflow.keras import layers
import tensorflow_addons as tfa


# In[2]:


import os, sys
root_dir, _ = os.path.split(os.getcwd())
script_dir = os.path.join(root_dir, 'scripts')
sys.path.append(script_dir)


# In[3]:


from hparams import hparams


# ## Invertible Convolution
# 
# The training boolean in the call method can be used to run the layer in reverse.
# 
# It could be worth investigating whether including the weight_norm wrapper of tensorflow addon can be used easily here and if it incurs significant improvements during training

# In[ ]:


class Invertible1x1Conv(layers.Layer):
  """
  Tensorflow 2.0 implementation of the inv1x1conv layer
  Deprecated
  """
  
  def __init__(self, filters, **kwargs):
    super(Invertible1x1Conv, self).__init__(**kwargs)
    self.kernel_size = filters
    self.activation = tf.keras.activations.get("linear")
    
  def build(self, input_shape):
    """This implementation assumes that the channel axis is last"""
    self.kernel = self.add_weight(
      shape=[1, self.filters, self.filters],
      initializer=tf.initializers.orthogonal(),
      trainable=True,
      dtype=self.dtype,
      name='kernel')
    
#   @tf.function  
  def call(self, inputs, training=True):
    """Training flag should be working now"""
    
    if training:
      
      # sign, log_det_weights = tf.linalg.slogdet(
      #   tf.cast(self.W, tf.float32))
      
      log_det_weights = tf.math.log(tf.math.abs(tf.linalg.det(
        tf.cast(self.kernel, tf.float32))))
      loss = - tf.cast(tf.reduce_sum(log_det_weights), 
                       dtype=self.dtype)
      # sign, log_det_weights = tf.linalg.slogdet(self.W)
  
      # loss = - tf.reduce_sum(log_det_weights)
      self.add_loss(loss)
      tf.summary.scalar(name='loss',
                       data=loss)
         
      output = tf.nn.conv1d(inputs, self.kernel, 
                            stride=1, padding='SAME')
    
    else:
      if not hasattr(self, 'kernel_inverse'):
        self.kernel_inverse = tf.cast(tf.linalg.inv(
          tf.cast(self.kernel, tf.float64)), dtype=self.dtype)
        
      return tf.nn.conv1d(inputs, self.kernel_inverse, 
                            stride=1, padding='SAME')   
  
  def get_config(self):
    config = super(Invertible1x1Conv, self).get_config()
    config.update(kernel_size = self.kernel_size)
    
    return config
      


# In[ ]:


class Inv1x1Conv(layers.Conv1D):
  """
  Tensorflow 2.0 implementation of the inv1x1conv layer 
  directly subclassing the tensorflow Conv1D layer
  """
  
  def __init__(self, filters, **kwargs):
    super(Inv1x1Conv, self).__init__(
      filters=filters,
      kernel_size=1,
      strides=1,
      padding='SAME',
      use_bias=False,
      kernel_initializer=tf.initializers.orthogonal(),
      activation="linear",
      **kwargs)
  
  def call(self, inputs, training=True):
    if training:
      sign, log_det_weights = tf.linalg.slogdet(
        tf.cast(self.kernel, tf.float32))
      loss = - tf.cast(tf.reduce_sum(log_det_weights), 
                       dtype=self.dtype)
      self.add_loss(loss)
      tf.summary.scalar(name='loss',
                       data=loss)
      return super(Inv1x1Conv, self).call(inputs)
      
    else:
      if not hasattr(self, 'kernel_inverse'):
        self.kernel_inverse = tf.cast(tf.linalg.inv(
          tf.cast(self.kernel, tf.float64)), dtype=self.dtype)
        
      return tf.nn.conv1d(inputs, self.kernel_inverse, 
                            stride=1, padding='SAME')


# ## Nvidia WaveNet Implementation
# Difference with the original implementations :
# WaveNet convonlution need not be causal. 
# No dilation size reset. 
# Dilation doubles on each layer
# 
# It could be worth investigating whether including the weight_norm wrapper of tensorflow addon incurs significant improvements during training

# In[ ]:


class WaveNetNvidia(layers.Layer):
  """
  Wavenet Block as defined in the WaveGlow implementation from Nvidia
  
  WaveNet convonlution need not be causal. 
  No dilation size reset. 
  Dilation doubles on each layer.
  """
  def __init__(self, n_in_channels, n_channels = 256, 
               n_layers = 12, kernel_size = 3, **kwargs):
    super(WaveNetNvidia, self).__init__(**kwargs)
    
    assert(kernel_size % 2 == 1)
    assert(n_channels % 2 == 0)
    
    self.n_layers = n_layers
    self.n_channels = n_channels
    self.n_in_channels = n_in_channels
    self.kernel_size = kernel_size
    
    self.in_layers = []
    self.res_skip_layers = []
    self.cond_layers = []
    
    self.start = tfa.layers.wrappers.WeightNormalization(
      layers.Conv1D(filters=self.n_channels,
                    kernel_size=1,
                    dtype=self.dtype,
                    name="start"))

    self.end = tfa.layers.wrappers.WeightNormalization(
      layers.Conv1D(filters=2 * self.n_in_channels,
                    kernel_size = 1,
                    kernel_initializer=tf.initializers.zeros(),
                    bias_initializer=tf.initializers.zeros(),
                    dtype=self.dtype,
                    name="end"))

    for index in range(self.n_layers):
      dilation_rate = 2 ** index
      
      in_layer = tfa.layers.wrappers.WeightNormalization(
        layers.Conv1D(filters=2 * self.n_channels,
                      kernel_size= self.kernel_size,
                      dilation_rate=dilation_rate,
                      padding="SAME",
                      dtype=self.dtype,
                      name="conv1D_{}".format(index)))    
      self.in_layers.append(in_layer)
      
      
      cond_layer = tfa.layers.wrappers.WeightNormalization(
        layers.Conv1D(filters = 2 * self.n_channels,
                      kernel_size = 1,
                      padding="SAME",
                      dtype=self.dtype,
                      name="cond_{}".format(index)))
      self.cond_layers.append(cond_layer)
      
      if index < self.n_layers - 1:
        res_skip_channels = 2 * self.n_channels
      else:
        res_skip_channels = self.n_channels
        
      res_skip_layer = tfa.layers.wrappers.WeightNormalization(
        layers.Conv1D(
          filters=res_skip_channels,
          kernel_size=1,
          dtype=self.dtype,
          name="res_skip_{}".format(index)))
      
      self.res_skip_layers.append(res_skip_layer)
      
    
  def call(self, inputs):
    """
    This implementatation does not require exposing a training boolean flag 
    as only the affine coupling behaviour needs reversing during
    inference.
    """
    audio_0, spect = inputs
    
    started = self.start (audio_0)   
    
    for index in range(self.n_layers):
      in_layered = self.in_layers[index](started)
      cond_layered = self.cond_layers[index](spect)
      
      half_tanh, half_sigmoid = tf.split(
        in_layered + cond_layered, 2, axis=2)
      half_tanh = tf.nn.tanh(half_tanh)
      half_sigmoid = tf.nn.sigmoid(half_sigmoid)
    
      activated = half_tanh * half_sigmoid
      
      res_skip_activation = self.res_skip_layers[index](activated)
      
      if index < (self.n_layers - 1):
        res_skip_activation_0, res_skip_activation_1 = tf.split(
          res_skip_activation, 2, axis=2)
        started = res_skip_activation_0 + started
        skip_activation = res_skip_activation_1
      else:
        skip_activation = res_skip_activation

      if index == 0:
        output = skip_activation
      else:
        output = skip_activation + output
        
    output = self.end(output)
    
    log_s, bias = tf.split(output, 2, axis=2)
    
    return output
  
  def get_config(self):
    config = super(WaveNetBlock, self).get_config()
    config.update(n_in_channels = self.n_in_channels)
    config.update(n_channels = self.n_channels)
    config.update(n_layers = self.n_layers)
    config.update(kernel_size = self.kernel_size)
  
    return config


# ## Custom Affine Coupling Layer
# 
# This layer does not have any trainable weights. It can be inverted by setting the training boolean to false.

# In[ ]:


class AffineCoupling(layers.Layer):
  """
  Invertible Affine Layer
  
  The inverted behaviour is obtained by setting the training boolean
  in the call method to false
  """
  
  def __init__(self, **kwargs):
    super(AffineCoupling, self).__init__(**kwargs)
    
  def call(self, inputs, training=None):
    
    audio_1, wavenet_output = inputs
    
    log_s, bias = tf.split(wavenet_output, 2, axis=2)
    
    if training:
      audio_1 = audio_1 * tf.math.exp(log_s) + bias
      loss = - tf.reduce_sum(log_s)
      self.add_loss(loss)
      tf.summary.scalar(name='loss', data=loss)
    else:
      audio_1 = (audio_1 - bias) *  tf.math.exp( - log_s)      
    
    return audio_1
  
  def get_config(self):
    config = super(AffineCoupling, self).get_config()
    
    return config


# ## WaveNet And Affine Coupling
# This block is a convenience block which has been defined to make it more straightforward to implement the WaveGlow model using the keras functional API. Note that affine coupling is the choice made in the original implementation of WaveGlow, but other choices are possible.

# In[ ]:


class WaveNetAffineBlock(layers.Layer):
  """
  Wavenet + Affine Layer
  Convenience block to provide a tidy model definition
  """
  def __init__(self, n_in_channels, n_channels = 256,
               n_layers = 12, kernel_size = 3, **kwargs):
    super(WaveNetAffineBlock, self).__init__(**kwargs)
    
    self.n_layers =  n_layers
    self.n_channels = n_channels
    self.n_in_channels = n_in_channels
    self.kernel_size = kernel_size
    
    self.wavenet = WaveNetNvidia(n_in_channels=n_in_channels,
                                 n_channels=n_channels,
                                 n_layers=n_layers,
                                 kernel_size=kernel_size,
                                 dtype=self.dtype)
    
    self.affine_coupling = AffineCoupling(dtype=self.dtype)
      
    
  def call(self, inputs, training=None):
    """
    training should be set to false to inverse affine layer
    """
    audio, spect = inputs
    audio_0, audio_1 = tf.split(audio, 2, axis=2)
    
    wavenet_output = self.wavenet((audio_0, spect))
    
    audio_1 = self.affine_coupling(
      (audio_1, wavenet_output), training=training)   
         
    audio = layers.Concatenate(axis=2) ([audio_0, audio_1])
    
    return audio
  
  def get_config(self):
    config = super(WaveNetBlock, self).get_config()
    config.update(n_in_channels = self.n_in_channels)
    config.update(n_channels = self.n_channels)
    config.update(n_layers = self.n_layers)
    config.update(kernel_size = self.kernel_size)
  
    return config


# ## Implementation of WeightNormalisation
# This implementation has been copied from the tensorflow-addons library

# In[ ]:


class WeightNormalization(tf.keras.layers.Wrapper):
    """This wrapper reparameterizes a layer by decoupling the weight's
    magnitude and direction.
    This speeds up convergence by improving the
    conditioning of the optimization problem.
    Weight Normalization: A Simple Reparameterization to Accelerate
    Training of Deep Neural Networks: https://arxiv.org/abs/1602.07868
    Tim Salimans, Diederik P. Kingma (2016)
    WeightNormalization wrapper works for keras and tf layers.
    ```python
      net = WeightNormalization(
          tf.keras.layers.Conv2D(2, 2, activation='relu'),
          input_shape=(32, 32, 3),
          data_init=True)(x)
      net = WeightNormalization(
          tf.keras.layers.Conv2D(16, 5, activation='relu'),
          data_init=True)(net)
      net = WeightNormalization(
          tf.keras.layers.Dense(120, activation='relu'),
          data_init=True)(net)
      net = WeightNormalization(
          tf.keras.layers.Dense(n_classes),
          data_init=True)(net)
    ```
    Arguments:
      layer: a layer instance.
      data_init: If `True` use data dependent variable initialization
    Raises:
      ValueError: If not initialized with a `Layer` instance.
      ValueError: If `Layer` does not contain a `kernel` of weights
      NotImplementedError: If `data_init` is True and running graph execution
      
      Deprecated
    """

    def __init__(self, layer, data_init=True, **kwargs):
        super(WeightNormalization, self).__init__(layer, **kwargs)
        self.data_init = data_init
        self._initialized = False
        self._track_trackable(layer, name='layer')

    def build(self, input_shape):
        """Build `Layer`"""
        input_shape = tf.TensorShape(input_shape).as_list()
        self.input_spec = tf.keras.layers.InputSpec(shape=input_shape)

        if not self.layer.built:
            self.layer.build(input_shape)

            if not hasattr(self.layer, 'kernel'):
                raise ValueError('`WeightNormalization` must wrap a layer that'
                                 ' contains a `kernel` for weights')

            # The kernel's filter or unit dimension is -1
            self.layer_depth = int(self.layer.kernel.shape[-1])
            self.kernel_norm_axes = list(
                range(self.layer.kernel.shape.rank - 1))

            self.v = self.layer.kernel
            self.g = self.add_variable(
                name="g",
                shape=(self.layer_depth,),
                initializer=tf.keras.initializers.get('ones'),
                dtype=self.layer.kernel.dtype,
                trainable=True)

        super(WeightNormalization, self).build()

    @tf.function
    def call(self, inputs):
        """Call `Layer`"""
        if not self._initialized:
            self._initialize_weights(inputs)

        self._compute_weights()  # Recompute weights for each forward pass
        output = self.layer(inputs)
        return output

    def compute_output_shape(self, input_shape):
        return tf.TensorShape(
            self.layer.compute_output_shape(input_shape).as_list())

    def _compute_weights(self):
        """Generate normalized weights.
        This method will update the value of self.layer.kernel with the
        normalized value, so that the layer is ready for call().
        """
        with tf.name_scope('compute_weights'):
            self.layer.kernel = tf.nn.l2_normalize(
                self.v, axis=self.kernel_norm_axes) * self.g

    def _initialize_weights(self, inputs):
        """Initialize weight g.
        The initial value of g could either from the initial value in v,
        or by the input value if self.data_init is True.
        """
        if self.data_init:
            self._data_dep_init(inputs)
        else:
            self._init_norm()
        self._initialized = True

    def _init_norm(self):
        """Set the weight g with the norm of the weight vector."""
        with tf.name_scope('init_norm'):
            flat = tf.reshape(self.v, [-1, self.layer_depth])
            self.g.assign(
                tf.reshape(tf.linalg.norm(flat, axis=0), (self.layer_depth,)))

    def _data_dep_init(self, inputs):
        """Data dependent initialization."""

        with tf.name_scope('data_dep_init'):
            # Generate data dependent init values
            existing_activation = self.layer.activation
            self.layer.activation = None
            x_init = self.layer(inputs)
            data_norm_axes = list(range(x_init.shape.rank - 1))
            m_init, v_init = tf.nn.moments(x_init, data_norm_axes)
            scale_init = 1. / tf.math.sqrt(v_init + 1e-10)

        # Assign data dependent init values
        self.g = self.g * scale_init
        if hasattr(self.layer, 'bias'):
            self.layer.bias = -m_init * scale_init
        self.layer.activation = existing_activation
        

    def get_config(self):
        config = {'data_init': self.data_init}
        base_config = super(WeightNormalization, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


# ## Custom Implementation of WeightNormalisedInvertible1x1Convolution

# In[78]:


class Inv1x1ConvWeightNorm(layers.Conv1D):
  
  
  def __init__(self, filters, **kwargs):
    super(Inv1x1ConvWeightNorm, self).__init__(
      filters=filters,
      kernel_size=1,
      strides=1,
      padding='SAME',
      use_bias=False,
      kernel_initializer=tf.initializers.orthogonal(),
      activation="linear",
      **kwargs)
    self._initialized = False
    
  def build(self, input_shape):
    super(Inv1x1ConvWeightNorm, self).build(input_shape)
    
    self.layer_depth = self.filters
    self.kernel_norm_axes = [0, 1]
      
    self.v = self.kernel
    self.g = self.add_variable(
        name="g",
        shape=self.layer_depth,
        initializer=tf.keras.initializers.get('ones'),
        dtype=self.dtype,
        trainable=True)
    
    flat = tf.squeeze(self.v, axis=0)
    self.g.assign(tf.linalg.norm(flat, axis=0))
  
  def call(self, inputs, training=True):
    if training:
      self.kernel = tf.nn.l2_normalize(
        self.v, axis=self.kernel_norm_axes) * self.g
      
      sign, log_det_weights = tf.linalg.slogdet(
        tf.cast(self.kernel, tf.float32))
      loss = - tf.cast(tf.reduce_sum(log_det_weights), 
                       dtype=self.dtype)
      self.add_loss(loss)
      tf.summary.scalar(name='loss',
                       data=loss)
      return super(Inv1x1ConvWeightNorm, self).call(inputs)
      
    else:
      if not hasattr(self, 'kernel_inverse'):
        self.kernel_inverse = tf.cast(tf.linalg.inv(
          tf.cast(self.kernel, tf.float64)), dtype=self.dtype)
        
      return tf.nn.conv1d(inputs, self.kernel_inverse, 
                            stride=1, padding='SAME')

