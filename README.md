# Waveglow in Tensorflow2
![GitHub](https://img.shields.io/github/license/vatj/waveglow-tensorflow2)
Custom implementation of the [Nvidia WaveGlow model by Prender et al.](https://arxiv.org/abs/1811.00002) using Tensorflow 2.0. You can find audio samples [here](http://files.tcm.phy.cam.ac.uk/~vatj2/waveglowTensorflow2.html).

## Quickstart

Download repository. Create a virtualenv and install the required packages. Create default directories:

```shell
git clone git@github.com:vatj/waveglow-tensorflow2.git
cd waveglow-tensorflow2
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
mkdir data logs checkpoints data/float32 logs/float32 checkpoints/float32
```

Set a path to dowload the LJSpeech Dataset in the scripts/hparams.py configuration file by modifying the data_dir entry and the floating point precision to use for training by editing the ftype entry. Run data preprocessing script (alternatively, one can run the full notebook in jupyter). Note that preprocessing will need to be run again to train with a different float type. Run the training script.

```shell
python scripts/raw_ljspeech_to_tfrecords.py
python scripts/training_main.py
```

Use tensorboard in a notebook to monitor training or run tensorboard directly from the command line

```shell
jupyter-notebook notebooks/control_tensorboard.ipynb
# or
tensorboard --log-dir ./logs/float32
```

## TODOS

- [ ] Enable cloud TPU support
- [ ] Fixing half-precision issues. It seems like computing determinant in half precision is unstable in current implementation.
- [ ] Add metric to the training loop e.g. mean loss over epoch
