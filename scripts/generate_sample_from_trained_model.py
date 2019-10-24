#!/usr/bin/env python
# coding: utf-8

# # Generate Long Audio  Sample from trained model

# ## Boilerplate

# In[1]:


import os, sys
root_dir, _ = os.path.split(os.getcwd())
script_dir = os.path.join(root_dir, 'scripts')
sys.path.append(script_dir)
os.environ['CUDA_VISIBLE_DEVICES'] = '0'


# In[2]:


import tensorflow as tf
print("GPU Available: ", tf.test.is_gpu_available())


# In[3]:


tf.keras.backend.clear_session()
# tf.keras.backend.set_floatx('float16')
# tf.debugging.set_log_device_placement(True)
gpus = tf.config.experimental.list_physical_devices('GPU')


# In[4]:


from hparams import hparams
from waveglow_model import WaveGlow
import training_utils as utils
import random
import pathlib
import pandas as pd
import numpy as np
import IPython.display as ipd


# In[19]:


show_audio = False
save_audio = True


# ## Load Long Audio Dataset

# In[5]:


test_dataset = utils.load_single_file_tfrecords(
  record_file=os.path.join(hparams['tfrecords_dir'], hparams['test_file']))
test_dataset = test_dataset.batch(
  hparams['train_batch_size'])


# ## Load long samples

# In[6]:


data_root_orig = tf.keras.utils.get_file(origin='https://data.keithito.com/data/speech/LJSpeech-1.1.tar.bz2',
                                         fname='LJSpeech-1.1', untar=True, cache_dir=hparams['data_dir'])
data_root = pathlib.Path(data_root_orig)


# In[7]:


# data_root = pathlib.Path(hparams['data_dir'])
all_sound_paths = list(data_root.glob('*/*'))
all_sound_paths = [str(path) for path in all_sound_paths]

random.seed(a=1234)
random.shuffle(all_sound_paths)


# ## Load preprocessed long audio split mel spectrograms

# In[8]:


long_audio_record_file = os.path.join(hparams['tfrecords_dir'], hparams['long_audio_file'])
long_audio_dataset = utils.load_long_audio_tfrecords(long_audio_record_file).batch(hparams['train_batch_size'])


# ## Instantiate model

# In[9]:


myWaveGlow = WaveGlow(hparams=hparams, name='myWaveGlow')
optimizer = utils.get_optimizer(hparams=hparams)


# ## Model Checkpoints : Initialise or Restore

# In[10]:


checkpoint = tf.train.Checkpoint(step=tf.Variable(0), 
                                 optimizer=optimizer, 
                                 net=myWaveGlow)

manager_checkpoint = tf.train.CheckpointManager(
  checkpoint, 
  directory=hparams['checkpoint_dir'],
  max_to_keep=hparams['max_to_keep'])

checkpoint.restore(manager_checkpoint.latest_checkpoint)

if manager_checkpoint.latest_checkpoint:
  tf.print('Restored from {checkpoint_dir}'.format(**hparams))
else:
  raise ValueError('Fetch a valid checkpoint!')


# In[11]:


batched_long_audios = []
for x_train in long_audio_dataset:
  batched_long_audios.append(myWaveGlow.infer(x_train['mel']))


# In[17]:


audios = dict()
originals = dict()

for x_train, samples in zip(long_audio_dataset, batched_long_audios):
  splits = tf.unique_with_counts(x_train['path'])
  long_audios = [audio for audio in tf.split(samples, splits.count)]
  for index, path in enumerate(splits.y.numpy()):
    if path.decode('utf-8') in audios.keys():
      audios[path.decode('utf-8')] = tf.concat([audios[path.decode('utf-8')], tf.reshape(long_audios[index], [-1])], axis=0)
    else:
      audios[path.decode('utf-8')] = tf.reshape(long_audios[index], [-1]) 
      signal = tf.io.read_file(path)
      original = np.squeeze(tf.audio.decode_wav(signal).audio.numpy())
      originals[path.decode('utf-8')] = original


# In[20]:


if show_audio:
  for original, audio in zip(originals.values(), audios.values()):
    print('original')
    ipd.display(ipd.Audio(original[:audio.shape[0]], rate=hparams['sample_rate']))
    print('generated')
    ipd.display(ipd.Audio(audio, rate=hparams['sample_rate']))


# In[23]:


if save_audio:
  for (path, original), audio in zip(originals.items(), audios.values()):
    print(path)
    _ , name = os.path.split(path)
    original_wav = tf.audio.encode_wav(tf.expand_dims(original[:audio.shape[0]], axis=1), sample_rate=hparams['sample_rate'])
    tf.io.write_file(filename=os.path.join(os.getcwd(), '..', 'data', 'audio_samples', 'original_' + name), contents=original_wav)
    audio_wav = tf.audio.encode_wav(tf.expand_dims(audio, axis=1), sample_rate=hparams['sample_rate'])
    tf.io.write_file(filename=os.path.join(os.getcwd(), '..', 'data', 'audio_samples', 'generated_' + name), contents=audio_wav)


# In[ ]:




