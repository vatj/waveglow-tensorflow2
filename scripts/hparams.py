#!/usr/bin/env python
# coding: utf-8

# # Hyperparameter dictionary

# In[1]:


import tensorflow as tf


# In[ ]:


hparams = dict()


# ## Waveglow parameters

# In[ ]:


# Number of flow blocks/composite layers in the model. Each flow is Invertible Convolution + Affine Coupling Layer to Wavenet
hparams['n_flows'] = 12
# Number of initial channels in invertible convolution
hparams['n_group'] = 8
# Number of flow layers between each channel reduction
hparams['n_early_every'] = 4
# Number of channels to shave from audio sample every n_early_every
hparams['n_early_size'] = 2
# Upsamplig scale of the mel spectrogram
hparams['upsampling_size'] = 256 
# Gaussian mixture standard deviation
hparams['sigma'] = 1.0
# Hidden Channels
hparams["hidden_channels"] = 256


# ## Wavenet Parameters

# In[ ]:


# Number of composite layer in Wavenet acting first with separate Conv1D on audio and spectrogram
# followed by sigmoid and tanh activations. Last comes a skip_layer
hparams['n_layers'] = 8
# Kernel size of the Wavenet convolution, no causality restriction
hparams['kernel_size'] = 3
# Number of hidden channels in Wavenet composite layers
hparams['n_channels'] = 256
# Only half of the audio sample channels go through wavenet. Other half will be conditioned by its result 
# via affine coupling layer
hparams['n_in_channels'] = 4


# ## Preprocessing of audio samples from LJSpeech

# In[ ]:


# Window of the short-time Fourier transform
hparams['fft_size'] = 1024
# Window shift of the short-time Fourier transform
hparams['hop_size'] = 256
# Crop audio samples to length 16000 (~5-10% of original)
hparams['segment_length'] = 16000
# Number of band in mel_spectrum. Becomes number of channel for spectrogram
hparams['mel_channels'] = 80
# Wav file sampling rate
hparams['sample_rate'] = 22050
# Cut-off frequencies for the fast Fourier transform
hparams['fmin'] = 0.0
hparams['fmax'] = 8000.0


# ## Machinery Details

# In[ ]:


# Floating precision. Float16 is not supported on cpus
hparams['ftype'] = tf.float32
# Batch size for training
hparams['train_batch_size'] = 4
# Learning rate, set to range(1e-3, 1e-4) for Adam and 1.0 for AdaDelta. Learning rate scheduler not supported yet
hparams['learning_rate'] = 3e-4
# Number of epochs to iterate over. Might be replaced by a number of training step in the future
hparams['epochs'] = 10
# Buffer size for shuffling
hparams['buffer_size'] = 12
# Optimizer, either Adam or AdaDelta. If AdaDelta, learning_rate = 1.0. 
# Added experimental tensorflow wrapper to support tf.float16
hparams['optimizer'] = "Adam" 


# In[ ]:


# Save model every number of step
hparams['save_model_every'] = 10
# Save audio samples every number of step
hparams['save_audio_every'] = 5
# Number of checkpoint files to keep
hparams['max_to_keep'] = 3 


# ## Generate tfrecords files

# In[ ]:


# Split training data in n_shards tfrecords files
hparams['n_shards'] = 12 
# Number of excluded example from training set to generate audio samples at each epochs for quality assessment
hparams['n_eval_samples'] = 20 
# Number of audio samples to run
hparams['n_test_samples'] = 80
# Training tfrecords common filename
hparams['train_files'] = 'ljs_train'
# Evaluation tfrecords filename
hparams['eval_file'] = 'ljs_eval.tfrecords'
# Test tfrecords filename
hparams['test_file'] = 'ljs_test.tfrecords'


# ## Path

# In[ ]:


# Raw data directory
hparams['data_dir'] = "/home/jupyter/.keras/datasets/LJSpeech-1.1"
# Tfrecords directory. Use different directories for float32 and float16 to avoid rerun of preprocessing
hparams['tfrecords_dir'] = "/home/jupyter/waveglow-tensorflow2/data/float32/"
# Log directory for tf.summary and tensorboard
hparams['log_dir'] = "/home/jupyter/waveglow-tensorflow2/logs/test/weight_norm_float32m"
# Checkpoint directory to save and restore model
hparams['checkpoint_dir'] = "/home/jupyter/waveglow-tensorflow2/checkpoints/test/weight_norm_float32m" 


# In[ ]:


# Legacy or Not implemented
# Not sure if it is used anywhere, I think it is equivalent the number of channels in Wavenet
# hparams['train_steps'] = 100  # Not implemented

