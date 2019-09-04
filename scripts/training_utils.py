#!/usr/bin/env python
# coding: utf-8

# # Useful functions (needs refactoring)

# In[1]:


import tensorflow as tf


# In[2]:


import os, sys
root_dir, _ = os.path.split(os.getcwd())
script_dir = os.path.join(root_dir, 'scripts')
sys.path.append(script_dir)


# In[3]:


import pprint


# In[4]:


from hparams import hparams


# ## Dataset loading

# In[5]:


sound_feature_description = {
    "wav": tf.io.FixedLenFeature([], tf.string),
    "mel": tf.io.FixedLenFeature([], tf.string)
}

def _parse_sound_function(example_proto):
  x = tf.io.parse_single_example(example_proto, sound_feature_description)
  x['wav'] = tf.io.parse_tensor(x['wav'], out_type=hparams['ftype'])
  x['mel'] = tf.io.parse_tensor(x['mel'], out_type=hparams['ftype'])  
  return x


# In[6]:


def load_single_file_tfrecords(record_file):
  raw_sound_dataset = tf.data.TFRecordDataset(record_file)
  parsed_sound_dataset = raw_sound_dataset.map(_parse_sound_function)
  return parsed_sound_dataset

def load_training_files_tfrecords(record_pattern):
  record_files = tf.data.TFRecordDataset.list_files(
    file_pattern=record_pattern)
  raw_sound_dataset = record_files.interleave(
    tf.data.TFRecordDataset,
    cycle_length=1,
    num_parallel_calls=tf.data.experimental.AUTOTUNE)
  parsed_sound_dataset = raw_sound_dataset.map(
    _parse_sound_function,
    num_parallel_calls=tf.data.experimental.AUTOTUNE)
  
  training_dataset = parsed_sound_dataset.shuffle(
    buffer_size=hparams['buffer_size']).batch(
    hparams['train_batch_size']).prefetch(
    buffer_size=tf.data.experimental.AUTOTUNE)
  
  return training_dataset


# ## Validation Step

# In[7]:


def eval_step(eval_dataset, waveGlow, hparams, step):
  if tf.equal(step, 0):
    for sample, x in eval_dataset.enumerate():
      x['wav'] = tf.cast(x['wav'], dtype=tf.float32)
      tf.summary.audio(name='original_{}'.format(sample),
                       data=tf.expand_dims(x['wav'], axis=2),
                       sample_rate=hparams['sample_rate'],
                       max_outputs=hparams['train_batch_size'],
                       encoding='wav',
                       step=step)
    tf.summary.text(name='hparams',
                    data=pprint.pformat(hparams),
                    step=step)
    tf.summary.text(name="checkpoint_restore",
                    data="Initializing from scratch.",
                    step=step)
  else:
    for sample, x in eval_dataset.enumerate():
      eval_samples = waveGlow.infer(x['mel'])
      eval_samples = tf.cast(eval_samples, dtype=tf.float32)
      tf.summary.audio(name='generated_{}_at_{}'.format(sample, step),
                       data=tf.expand_dims(eval_samples, axis=2),
                       sample_rate=hparams['sample_rate'],
                       max_outputs=hparams['train_batch_size'],
                       encoding='wav',
                       step=step)


# ## Optimizer compatibility with tf.float16

# In[8]:


def get_optimizer(hparams):
  """
  Return optimizer instance based on hparams
  
  Wrap the optimizer to avoid underflow if ftype=tf.float16
  """
  if hparams['optimizer'] == "Adam":
    optimizer = tf.keras.optimizers.Adam(
      learning_rate=hparams["learning_rate"])
  elif hparams['optimizer'] == "Adadelta":
    assert(hparams["learning_rate"] == 1.0), "Set learning_rate to 1.0"
    optimizer = tf.keras.optimizers.Adadelta(
      learning_rate=hparams['learning_rate'])
  else:
    raise "Supported Optimizer is either Adam or Adagrad"
    
  if hparams["ftype"] == tf.float16:
    return tf.train.experimental.enable_mixed_precision_graph_rewrite(
      optimizer, "dynamic")
  else:
    return optimizer

