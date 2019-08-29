# Waveglow in Tensorflow2
[License Badge]()
Custom implementation of the [Nvidia WaveGlow model by Prender et al.](https://arxiv.org/abs/1811.00002) using Tensorflow 2.0.

## Quickstart

Download repository. Create a virtualenv and install the required packages. Create default directories:

```shell
git clone blabla
cd blabla
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
mkdir data logs checkpoints
```

Set a path to dowload the LJSpeech Dataset in the scripts/hparams.py configuration file by modifying the data_dir entry and the floating point precision to use for training by editing the ftype entry. Run data preprocessing script (alternatively, one can run the full notebook in jupyter). Note that preprocessing will need to be run again to train with a different float type. Run the training script.

```shell
python -m scripts/data_preprocess.py
python -m scripts/training.py
```

Use tensorboard in a notebook to monitor training or run tensorboard directly from the command line

```shell
jupyter-notebook notebooks/tensorboard.ipynb
# or
tensorboard --log-dir ./logs
```

## TODOS

- [ ] Enable cloud TPU support
- [ ] Fixing half-precision issues. It seems like computing determinant in half precision is unstable in current implementation.
- [ ] Add metric to the training loop e.g. mean loss over epoch
- [ ] Add a notebook with a couple of iteration to enable profiling and graph of the train step autograph
- [ ] Hyperparameters need to be commented further
- [ ] Train the model and add link to audio samples

## Supported

### Training :
- [x] Train on subset of LJSpeech Dataset ()
- [x] Train on full LJSpeech Dataset (limited number of steps on CPUs)
- [x] Working training loop for subclass waveglow
- [x] Training full LJSpeech Dataset GPU using Nvidia V100 GPUS float32

### Tensorboard :
- [x] Add minimum tensorboard support to observe losses evolution during training
- [x] Added generated audio samples every epoch
- [x] Redirected notebook print to tf.summary.text


### Datasets :
- [x] Make tfrecords file which contains a subset of LJ_Speech dataset
- [x] Split the dataset into training, validation and testing 

### Custom Layers :
- [x] Change the tensorflow Op layers into built-in keras layers (e.g. tf.concat to layers.Concat)
- [x] Remove the reverse flag and use the training flag instead.
- [x] Modularize WaveNetAffineBlock as WaveNetNvidia + AffineCoupling Custom Layers


### Subclass_waveglow file :
- [x] Needs to be refactored. Currently contains a waveglow implementation which subclass keras.Model as well as custom layer implementations 
- [x] The infer method is not functional on this implementation. Should be done after switching from reverse to training flag on the custom layers. Corrected and tested
- [x] Change the tensorflow Op layers into built-in keras layers (e.g. tf.concat to layers.Concat)


### Hyper Parameters :
- [x] Focus on implementation of the hparams['ftype'] = tf.float16. Nvidia V100 reaches 125 TFLOPS instead of 15 with tf.float32


