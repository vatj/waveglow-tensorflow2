#!/usr/bin/env python
# coding: utf-8

# In[1]:


import tensorflow as tf


# In[2]:


import os, sys
root_dir, _ = os.path.split(os.getcwd())
script_dir = os.path.join(root_dir, 'scripts')
sys.path.append(script_dir)
sys.path.append('home/phd/miniconda3/')
from hparams import hparams


# In[3]:


get_ipython().run_line_magic('load_ext', 'tensorboard')


# In[4]:


from tensorboard import notebook


# In[5]:


notebook.list()


# In[6]:


log_dir, __ = os.path.split(hparams['log_dir'])


# In[ ]:


get_ipython().system('/home/phd/miniconda3/envs/tf2rc/bin/tensorboard --logdir $log_dir --port=6081')


# In[11]:





# In[ ]:




