#!/usr/bin/env python
# coding: utf-8

# # Load sounds with tf.data

# ## Setup

# In[1]:


import tensorflow as tf
import pathlib
import random
import os, sys
import math
import numpy as np


# In[2]:


root_dir, _ = os.path.split(os.getcwd())
script_dir = os.path.join(root_dir, 'scripts')
sys.path.append(script_dir)


# In[3]:


from hparams import hparams


# In[4]:


data_root_orig = tf.keras.utils.get_file(origin='https://data.keithito.com/data/speech/LJSpeech-1.1.tar.bz2',
                                         fname='LJSpeech-1.1', untar=True, cache_dir=hparams['data_dir'])

data_root = pathlib.Path(data_root_orig)


# In[5]:


# data_root = pathlib.Path(hparams['data_dir'])
all_sound_paths = list(data_root.glob('*/*'))
all_sound_paths = [str(path) for path in all_sound_paths]

random.seed(a=1234)
random.shuffle(all_sound_paths)


# ### Load and Pre-process wav files helpers

# In[6]:


def load_and_preprocess_wav_file(sound_path, hparams):
    sound = tf.io.read_file(sound_path)   
    return preprocess_wav_file(sound, hparams)

def preprocess_wav_file(sound, hparams):
    '''
    Read wav file and compute mel spectrogram
    '''
    signal = tf.squeeze(tf.audio.decode_wav(sound).audio)
    max_start = signal.shape[0] - hparams['segment_length']
    start = random.randrange(0, max_start)
    sound_tensor = signal[start:start+hparams['segment_length']]

    mel = compute_mel_spectrogram(sound_tensor, hparams)
    
    sound_tensor = tf.cast(sound_tensor, dtype=hparams['ftype'])
    mel = tf.cast(mel, dtype=hparams['ftype'])
    
    return sound_tensor, mel


# In[7]:


def compute_mel_spectrogram(sound_tensor, hparams):
    '''
    Compute mel spectrogram from sound tensor
    '''
    
    stft = tf.signal.stft(sound_tensor,
                          frame_length=hparams['fft_size'],
                          frame_step=hparams['hop_size'],
                          fft_length=hparams['fft_size'],
                          pad_end=True)
    
    magnitude = tf.abs(stft)

    linear_to_mel_weight_matrix = tf.signal.linear_to_mel_weight_matrix(
      hparams['mel_channels'], 
      magnitude.shape[-1],
      hparams['sample_rate'], 
      hparams['fmin'],
      hparams['fmax'])

    # Mel Spectrogram
    mel = tf.tensordot(magnitude, linear_to_mel_weight_matrix, 1)
    mel = tf.math.log(tf.maximum(mel, 1e-5)) # log scaling with clamping
    mel = tf.cast(mel, dtype=hparams['ftype'])
    
    return mel


# ### Serialize function and proto tf.Example

# In[8]:


# The following functions can be used to convert a value to a type compatible
# with tf.Example.

def _bytes_feature(value):
  """Returns a bytes_list from a string / byte."""
  if isinstance(value, type(tf.constant(0))):
    value = value.numpy() # BytesList won't unpack a string from an EagerTensor.
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def _float_feature(value):
  """Returns a float_list from a float / double."""
  return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))

def _int64_feature(value):
  """Returns an int64_list from a bool / enum / int / uint."""
  return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


# In[9]:


def sound_example(sound_path, hparams):
  '''
  Creates a tf.Example message from wav, mel
  '''

  wav, mel = load_and_preprocess_wav_file(sound_path, hparams)

  features = {
      "wav": _bytes_feature(tf.io.serialize_tensor(wav)),
      "mel": _bytes_feature(tf.io.serialize_tensor(mel))
  }

  return tf.train.Example(
      features=tf.train.Features(feature=features))


# ## Write TFRecord File for validation and testing

# In[10]:


# Create iterator to avoid writing any sample twice
path_ds = iter(tf.data.Dataset.from_tensor_slices(all_sound_paths))


# In[11]:


def single_tfrecords_writer(path_ds, record_file, n_samples, hparams):
    with tf.io.TFRecordWriter(record_file) as writer:
        for path, sample in zip(path_ds, range(n_samples)):
            tf_example = sound_example(path, hparams)
            writer.write(tf_example.SerializeToString())


# In[12]:


# Validation Samples
record_file = hparams['tfrecords_dir'] + hparams['eval_file']
sample = hparams['n_eval_samples']
single_tfrecords_writer(path_ds, record_file, sample, hparams)


# In[13]:


# Test Samples
record_file = hparams['tfrecords_dir'] + hparams['test_file']
sample = hparams['n_test_samples']
single_tfrecords_writer(path_ds, record_file, sample, hparams)


# ## Split Training Dataset in TFRecords Shards

# In[14]:


sample = len(all_sound_paths) - hparams['n_eval_samples'] - hparams['n_test_samples']
sample_per_shard = math.ceil(sample / hparams['n_shards'])
print(sample, sample_per_shard)


# In[15]:


for idx_shard in range(hparams['n_shards']):
  print("Currently saving {} samples in : ".format(sample_per_shard))
  fname = hparams['train_files'] +    '_{}_of_{}.tfrecords'.format(idx_shard, hparams['n_shards'] - 1)
  current_path = os.path.join(hparams['tfrecords_dir'], fname)
  print(current_path)
  with tf.io.TFRecordWriter(current_path) as writer:        
    for path, sample in zip(path_ds, range(sample_per_shard)):            
      tf_example = sound_example(tf.constant(path), hparams)
      writer.write(tf_example.SerializeToString())


# In[ ]:




