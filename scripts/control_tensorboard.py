#!/usr/bin/env python
# coding: utf-8

# In[1]:


import tensorflow as tf


# In[2]:


import os, sys
root_dir, _ = os.path.split(os.getcwd())
script_dir = os.path.join(root_dir, 'scripts')
sys.path.append(script_dir)
from hparams import hparams


# In[3]:


get_ipython().run_line_magic('load_ext', 'tensorboard')


# In[4]:


from tensorboard import notebook


# In[9]:


notebook.list()


# In[14]:


log_dir, __ = os.path.split(hparams['log_dir'])


# In[16]:


get_ipython().run_line_magic('tensorboard', '--logdir $log_dir --port=6081')


# In[ ]:




