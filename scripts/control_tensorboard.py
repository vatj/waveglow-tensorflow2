#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import tensorflow as tf


# In[ ]:


import os, sys
root_dir, _ = os.path.split(os.getcwd())
script_dir = os.path.join(root_dir, 'scripts')
sys.path.append(script_dir)
import hparams


# In[ ]:


get_ipython().run_line_magic('load_ext', 'tensorboard')


# In[ ]:


from tensorboard import notebook


# In[ ]:


notebook.list()


# In[ ]:


# %tensorboard --logdir ../logs/functional_test/ --port=6006


# In[ ]:


get_ipython().run_line_magic('tensorboard', '--logdir ../logs/test --port=6062')


# In[ ]:





# In[ ]:




