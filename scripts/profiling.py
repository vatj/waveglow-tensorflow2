#!/usr/bin/env python
# coding: utf-8

# # Train WaveGlow Model with custom training step

# ## Boilerplate Import

# In[1]:


import tensorflow as tf
print("GPU Available: ", tf.test.is_gpu_available())


# In[2]:


tf.keras.backend.clear_session()


# In[3]:


import os, sys
root_dir, _ = os.path.split(os.getcwd())
script_dir = os.path.join(root_dir, 'scripts')
sys.path.append(script_dir)
from datetime import datetime


# In[4]:


from hparams import hparams
from waveglow_model import WaveGlow
import training_utils as utils


# ## Tensorboard logs setup

# In[5]:


log_dir = hparams['log_dir'] + 'test/profiler'
file_writer = tf.summary.create_file_writer(log_dir)
file_writer.set_as_default()


# ## Load Validation and Training Dataset

# In[6]:


validation_dataset = utils.load_single_file_tfrecords(
  record_file=hparams['tfrecords_dir'] + hparams['eval_file'])
validation_dataset = validation_dataset.batch(
  hparams['train_batch_size'])


# In[7]:


training_dataset = utils.load_training_files_tfrecords(
  record_pattern=hparams['tfrecords_dir'] + hparams['train_files'] + '*')


# ## Instantiate model and optimizer

# In[8]:


myWaveGlow = WaveGlow(hparams=hparams, name='myWaveGlow')

optimizer = utils.get_optimizer(hparams=hparams)


# ## Model Checkpoints : Initialise or Restore

# In[9]:


checkpoint = tf.train.Checkpoint(step=tf.Variable(0), 
                                 optimizer=optimizer, 
                                 net=myWaveGlow)

manager_checkpoint = tf.train.CheckpointManager(
  checkpoint, 
  directory=hparams['checkpoint_dir'] + "test/weight_norm",
  max_to_keep=hparams['max_to_keep'])

checkpoint.restore(manager_checkpoint.latest_checkpoint)

if manager_checkpoint.latest_checkpoint:
  tf.summary.experimental.set_step(tf.cast(checkpoint.step, tf.int64))
  tf.summary.text(name="checkpoint_restore",
                  data="Restored from {}".format(manager_checkpoint.latest_checkpoint))
else:
  tf.summary.experimental.set_step(0)
  utils.eval_step(eval_dataset=validation_dataset,
                  waveGlow=myWaveGlow, hparams=hparams,
                  step=0)


# ## Training step autograph

# In[10]:


@tf.function
def train_step(step, x_train, waveGlow, hparams, optimizer):
  tf.summary.experimental.set_step(step=step)
  with tf.GradientTape() as tape:
    outputs = waveGlow(x_train, training=True)
    total_loss = waveGlow.total_loss(outputs=outputs)

  grads = tape.gradient(total_loss, 
                        myWaveGlow.trainable_weights)
  optimizer.apply_gradients(zip(grads, 
                                myWaveGlow.trainable_weights))


# In[11]:


def custom_training(waveGlow, hparams, optimizer, 
                    checkpoint, manager_checkpoint):
  step = tf.cast(checkpoint.step, tf.int64)
  
  for epoch in tf.range(hparams['epochs']):
    tf.summary.text(name='epoch',
                    data='Start epoch {}'.format(epoch.numpy()) +\
                    'at ' + datetime.now().strftime("%Y%m%d-%H%M%S"),
                    step=step)
    
    for step, x_train in zip(tf.range(3), training_dataset):
      if step == 2:
        tf.python.eager.profiler.start()
        
      train_step(step=step,
                 x_train=x_train,
                 waveGlow=waveGlow,
                 hparams=hparams,
                 optimizer=optimizer)
      
      if tf.equal(step % hparams['save_model_every'], 0):
        save_path = manager_checkpoint.save()
        tf.summary.text(name='save_checkpoint',
                        data="Saved checkpoint in" + save_path,
                        step=step)
        
      if tf.equal(step % hparams['save_audio_every'], 0):
        utils.eval_step(eval_dataset=validation_dataset,
                        waveGlow=waveGlow, hparams=hparams,
                        step=step)
      if step == 2:
        tf.python.eager.profiler.stop()
    
      checkpoint.step.assign_add(1)


# In[ ]:


custom_training(waveGlow=myWaveGlow, 
                hparams=hparams, 
                optimizer=optimizer,
                checkpoint=checkpoint,
                manager_checkpoint=manager_checkpoint)


# In[ ]:




