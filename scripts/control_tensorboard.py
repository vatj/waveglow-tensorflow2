#!/usr/bin/env python
# coding: utf-8

# In[1]:


import tensorflow as tf


# In[9]:


import os, sys
root_dir, _ = os.path.split(os.getcwd())
script_dir = os.path.join(root_dir, 'scripts')
sys.path.append(script_dir)
from hparams import hparams


# In[10]:


get_ipython().run_line_magic('load_ext', 'tensorboard')


# In[11]:


from tensorboard import notebook


# In[17]:


notebook.list()


# In[13]:


# %tensorboard --logdir ../logs/functional_test/ --port=6006


# In[14]:


log_dir = os.path.join(hparams['log_dir'], 'test')


# In[20]:


get_ipython().run_line_magic('tensorboard', '--logdir $log_dir --port=6080')


# In[18]:





# In[ ]:




