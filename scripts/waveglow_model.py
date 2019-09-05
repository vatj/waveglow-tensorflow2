#!/usr/bin/env python
# coding: utf-8

# In[1]:


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
from custom_layers import Inv1x1Conv, WaveNetAffineBlock


# In[ ]:


class WaveGlow(tf.keras.Model):
  """
  Waveglow implementation using the Invertible1x1Conv custom layer and 
  the WaveNet custom block 
  Likely change to have a hyperparameter dict
  The init function needs to be adjusted as we don't need to specify
  input dimension here as far as I understand the new 2.0 standards
  """
  
  def __init__(self, hparams, **kwargs):
    super(WaveGlow, self).__init__(dtype=hparams['ftype'], **kwargs)
    
    assert(hparams['n_group'] % 2 == 0)
    self.n_flows = hparams['n_flows']
    self.n_group = hparams['n_group']
    self.n_early_every = hparams['n_early_every']
    self.n_early_size = hparams['n_early_size']
    self.upsampling_size = hparams['upsampling_size']
    self.hidden_channels = hparams['hidden_channels']
    self.mel_channels = hparams['mel_channels']
    self.hparams = hparams
    self.normalisation = self.hparams['train_batch_size'] * self.hparams['segment_length']

    self.waveNetAffineBlocks = []
    self.weightNormInv1x1ConvLayers = []
    
    self.upsampling = layers.UpSampling1D(size=self.upsampling_size,
                                          dtype=self.dtype)
      
    n_half = self.n_group // 2
    n_remaining_channels = self.n_group
    
    for index in range(self.n_flows):
      if ((index % self.n_early_every == 0) and (index > 0)):
        n_half -= self.n_early_size // 2
        n_remaining_channels -= self.n_early_size
        
    
      self.weightNormInv1x1ConvLayers.append(
        tfa.layers.wrappers.WeightNormalization(
          Inv1x1Conv(
            filters=n_remaining_channels,
            dtype=hparams['ftype'],
            name="newInv1x1conv_{}".format(index)),
          data_init=False,
          dtype=hparams['ftype']))
      
      self.waveNetAffineBlocks.append(
        WaveNetAffineBlock(n_in_channels=n_half, 
                     n_channels=hparams['n_channels'],
                     n_layers=hparams['n_layers'],
                     kernel_size=hparams['kernel_size'],
                     dtype=hparams['ftype'],
                     name="waveNetAffineBlock_{}".format(index)))
      
    self.n_remaining_channels = n_remaining_channels
    
    
  def call(self, inputs, training=None):
    """
    Evaluate model against inputs
    
    if training is false simply return the output of the infer method,
    which effectively run through the layers backward and invert them.
    Otherwise run the network in the training "direction".
    """
    
    if not training:
      return self.infer(inputs)
    
    audio, spect = inputs['wav'], inputs['mel']
    
    audio = layers.Reshape(
      target_shape = [self.hparams["segment_length"] // self.n_group,
                      self.n_group],
      dtype=self.dtype) (audio)
    
    # No reshape happening here, but enforce well defined rank
    # for spect tensor which is required for upsampling layer
    spect = layers.Reshape(
      target_shape = [63, self.mel_channels],
      dtype=self.dtype) (spect)
    
    spect = self.upsampling(spect)
    
    spect = layers.Cropping1D(
      cropping=(0, spect.shape[1] - hparams['segment_length']),
      dtype=self.dtype) (spect)

    spect = layers.Reshape(
      [self.hparams["segment_length"] // self.n_group, 
       self.mel_channels * self.n_group],
      dtype=self.dtype) (spect)
    
    output_audio = []
    n_remaining_channels = self.n_group
    
    for index in range(self.n_flows):
      if ((index % self.n_early_every == 0) and (index > 0)):
        n_remaining_channels -= hparams['n_early_size']
        
        audio = layers.Permute(dims=(2, 1), dtype=self.dtype) (audio)
        output_chunk = layers.Cropping1D(
          cropping=(0, n_remaining_channels),
          dtype=self.dtype) (audio)
        audio = layers.Cropping1D(
          cropping=(hparams['n_early_size'], 0),
          dtype=self.dtype) (audio)
        audio = layers.Permute(dims=(2, 1), dtype=self.dtype) (audio)
        output_chunk = layers.Permute(dims=(2, 1), 
                                      dtype=self.dtype) (output_chunk)
        output_audio.append(output_chunk)
        
        # output_audio.append(audio[:, :, :self.n_early_size])
        # audio = audio[:,:,self.n_early_size:]
        
      # No need to output log_det_W or log_s as added as loss in custom 
      # layers 
      audio = self.weightNormInv1x1ConvLayers[index](audio)
      audio = self.waveNetAffineBlocks[index]((audio, spect),
                                              training=True)     
      
    output_audio.append(audio)
    self.custom_logging()
    
    return layers.Concatenate(axis=2, dtype=self.dtype) (output_audio)
  
  def infer(self, spect, sigma=1.0):
    """
    Push inputs through network in reverse direction.
    Two key aspects:
    Layers in reverse order.
    Layers are inverted through exposed training boolean.
    """

    spect = layers.Reshape(
      target_shape=[63,
                    self.hparams['mel_channels']]) (spect)
    
    spect = self.upsampling(spect)
    spect = layers.Cropping1D(
      cropping=(0, spect.shape[1] - self.hparams['segment_length'])) (spect)
    
    spect = layers.Reshape(
      [self.hparams["segment_length"] // self.n_group, 
       self.mel_channels * self.n_group]) (spect)
    
    audio = tf.random.normal(
      shape = [spect.shape[0],
               self.hparams['segment_length'] // self.n_group, 
               self.n_remaining_channels],
      dtype = self.hparams['ftype'])

    audio *= sigma
    
    for index in reversed(range(self.n_flows)):
      
      audio = self.waveNetAffineBlocks[index] ((audio, spect),
                                               training=False)
      
      audio = self.weightNormInv1x1ConvLayers[index](audio, training=False)
      
      if ((index % self.n_early_every == 0) and (index > 0)):
        z = tf.random.normal(
          shape = [spect.shape[0],
                   self.hparams['segment_length'] // self.n_group, 
                   self.n_early_size],
          dtype = self.hparams['ftype'])
        audio = layers.Concatenate(axis=2)([z * sigma, audio])
        
    audio = layers.Reshape(
      target_shape=[self.hparams['segment_length']]) (audio)
        
    return audio
  
  def get_config(self):
    config = super(WaveGlow, self).get_config()
    config.update(hparams = hparams)
    
    return config
  
  def custom_logging(self):
    
    invconv_loss = tf.math.accumulate_n(
      [layer.losses[0] for layer in self.weightNormInv1x1ConvLayers])
    affine_loss =  tf.math.accumulate_n(
      [layer.losses[0] for layer in self.waveNetAffineBlocks])
    
    tf.summary.scalar(name='invertible_layers_aggregated',
                      data=invconv_loss / self.n_group)
    tf.summary.scalar(name='wavenet_layers_aggregated',
                      data= (affine_loss / self.normalisation))

    for index in range(self.n_flows):
      tf.summary.scalar(
        name='flow_{}'.format(index),
        data=self.weightNormInv1x1ConvLayers[index].losses[0] +\
          (self.waveNetAffineBlocks[index].losses[0]/self.normalisation))
      
      tf.summary.scalar(
        name='AffineNormalized_{}'.format(index),
        data=self.waveNetAffineBlocks[index].losses[0]/self.normalisation)
      
      tf.summary.scalar(
        name='inv1x1convNormalized_{}'.format(index),
        data=self.weightNormInv1x1ConvLayers[index].losses[0]/self.n_group)
      
  
  def total_loss(self, outputs):
  
    outputs_loss = tf.reduce_sum(outputs * outputs)       / (2 * self.hparams['sigma'] * self.hparams['sigma'])

    affine_loss =  tf.math.accumulate_n(
      [layer.losses[0] for layer in self.waveNetAffineBlocks])
    invconv_loss =  tf.math.accumulate_n(
      [layer.losses[0] for layer in self.weightNormInv1x1ConvLayers])

    total_loss = (outputs_loss + affine_loss) / self.normalisation
    total_loss += (invconv_loss / self.n_group)

    tf.summary.scalar(name='total_loss',
                      data=total_loss)
  
    return total_loss


# In[ ]:




