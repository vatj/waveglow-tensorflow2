#!/usr/bin/env python
# coding: utf-8

# # Generate Long Audio Sample records file

# ## Boilerplate

# In[ ]:


import os, sys
root_dir, _ = os.path.split(os.getcwd())
script_dir = os.path.join(root_dir, 'scripts')
sys.path.append(script_dir)
os.environ['CUDA_VISIBLE_DEVICES'] = '0'


# In[ ]:


import tensorflow as tf
print("GPU Available: ", tf.test.is_gpu_available())


# In[ ]:


tf.keras.backend.clear_session()
# tf.keras.backend.set_floatx('float16')
# tf.debugging.set_log_device_placement(True)
gpus = tf.config.experimental.list_physical_devices('GPU')


# In[ ]:


from hparams import hparams
from waveglow_model import WaveGlow
import training_utils as utils
import random
import pathlib
import math


# ## Load long samples

# In[ ]:


data_root_orig = tf.keras.utils.get_file(origin='https://data.keithito.com/data/speech/LJSpeech-1.1.tar.bz2',
                                         fname='LJSpeech-1.1', untar=True, cache_dir=hparams['data_dir'])

data_root = pathlib.Path(data_root_orig)


# In[ ]:


# data_root = pathlib.Path(hparams['data_dir'])
all_sound_paths = list(data_root.glob('*/*'))
all_sound_paths = [str(path) for path in all_sound_paths]

random.seed(a=1234)
random.shuffle(all_sound_paths)


# In[ ]:


def split_and_preprocess_wav_file(sound_path, hparams):
    '''
    Read wav file and compute mel spectrogram
    '''
    sound = tf.io.read_file(sound_path)
    signal = tf.squeeze(tf.audio.decode_wav(sound).audio)
    number_of_slices = math.floor(signal.shape[0] / hparams['segment_length'])
    sound_tensors = [signal[i*hparams['segment_length']:(i+1)*hparams['segment_length']] for i in range(0, number_of_slices - 1)]

    mels = compute_mel_spectrograms(sound_tensors, hparams)
    
    sound_tensors = [tf.cast(sound_tensor, dtype=hparams['ftype']) for sound_tensor in sound_tensors]
    mels = [tf.cast(mel, dtype=hparams['ftype']) for mel in mels]
    
    return [dict(wav=sound_tensor, mel=mel, path=sound_path, number_of_slices=number_of_slices) for sound_tensor, mel in zip(sound_tensors, mels)]
  
def compute_mel_spectrograms(sound_tensors, hparams):
  '''
  Compute mel spectrogram from all sound tensors
  '''
  mels = []
  for sound_tensor in sound_tensors:
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
    mels.append(mel)

  return mels


# ### Serialize function and proto tf.Example

# In[ ]:


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


# In[ ]:


def sound_example(features, hparams):
  '''
  Creates a tf.Example message from wav, mel
  ''' 
  
  features = {
    "wav": _bytes_feature(tf.io.serialize_tensor(features['wav'])),
    "mel": _bytes_feature(tf.io.serialize_tensor(features['mel'])),
    "path": _bytes_feature(tf.io.serialize_tensor(features['path'])),
    "number_of_slices": _bytes_feature(tf.io.serialize_tensor(features['number_of_slices']))
  }

  return tf.train.Example(
      features=tf.train.Features(feature=features))


# In[ ]:


path_ds = iter(tf.data.Dataset.from_tensor_slices(all_sound_paths))


# In[ ]:


def single_tfrecords_writer(path_ds, record_file, n_samples, hparams):
    with tf.io.TFRecordWriter(record_file) as writer:
        for path, sample in zip(path_ds, range(n_samples)):
            features = split_and_preprocess_wav_file(path, hparams)
            for feature in features:
              tf_example = sound_example(features=feature, hparams=hparams)
              writer.write(tf_example.SerializeToString())


# In[ ]:


record_file = os.path.join(hparams['tfrecords_dir'], hparams['long_audio_file'])
sample = 30
single_tfrecords_writer(path_ds, record_file, sample, hparams)


# In[ ]:




