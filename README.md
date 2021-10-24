# Final project for 'Reinforcement Learning' course, 2021, Skoltech
Reimplementation of the original paper 

> [Implementation Matters in Deep Policy Gradients: A Case Study on PPO and TRPO](https://arxiv.org/abs/2005.12729)

Team members:
* Alexey Larionov

### Repository overview
* 👉 [`train.py`](train.py) - an entry point for models training
* 👉 [`notebooks/training.ipynb`](notebooks/training.ipynb) - a quickstart
  Jupyter notebook for training or loading from a checkpoint
* [`configs/`](configs/) - YAML files with parameters of all experiments
* [`src/`](src/) - all the code of interest
* [`utils/`](utils/) - auxiliary code, unrelated to Reinforcement Learning per se
* 👉 [`materials/`](materials/) - images, plots, the presentation, the original paper file

### Requirements
A GPU is recommended to perform the experiments. You can use [Google
Colab](colab.research.google.com) with Jupyter notebooks provided in
[`notebooks/`](notebooks/) folder

Libraries required:

- [`pytorch`](http://pytorch.org/) for models training
- [`pytorch-lightning`](https://www.pytorchlightning.ai/) with [`LightningCLI`](https://pytorch-lightning.readthedocs.io/en/latest/common/lightning_cli.html#lightningcli) for CLI tools and low-boilerplate models training
- [`torchvision`](https://anaconda.org/pytorch/torchvision)
- [`numpy`](https://anaconda.org/anaconda/numpy) for data generation

Optional libraries:
- [`google-colab`](https://anaconda.org/conda-forge/google-colab) - if you want to mount a Google Drive to your Jupyter Notebook to store training artifacts
- [`gdown`](https://anaconda.org/conda-forge/gdown) - if you want to download checkpoints from Google Drive by ID
- [`tensorboard`](https://anaconda.org/conda-forge/tensorboard) and `tensorboardX` - if you want to view training/validation metrics of different experiments



<!TODO:
For convenience, a ready to use conda [environment](environment.yml) is provided. 
To create a new python environment with all the required packages, you can run:
```shell
conda env create -f environment.yml
conda activate dyconv
```
!>

### Reproduce training and inference

#### Using [`train.py`](train.py)

Choose one config from [`configs/`](configs/) folder and run:
```bash
python train.py --config configs/your_chosen_experiment.yaml
```
As a result, the `downloads/` and `logs/` folder will appear with external
downloaded files and logs of each distinct experiment respectively. Inside every
experiment's folder you can find results of all the runs of this very same experiment, namely
folders like `version_0/`, `version_1/`, etc, which in turn will contain the following:
- `config.yaml` - same parameters of the experiment as in `your_chosen_experiment.yaml`
- `events.out.tfevents...` file with logs of Tensorboard
- `checkpoints/` directory with the best epoch checkpoint and the lastest epoch ckeckpoint, use those inference of training resuming 

#### Using Jupyter Notebook [`notebooks/training.ipynb`](notebooks/training.ipynb)

The notebook begins with a section with parameters you can set. Any other
sections don't usually need any adjustments. After you "Run All" the notebook,
either a training will start (a new one, or resumed), or only the model weights
will be loaded (if you've chosen to `'load_model'`, as the notebook prescribes). 

Anyway after the notebook has been run completely, there will be a `model`
variable of type
[`pytorch_lightning.LightningModule`](https://pytorch-lightning.readthedocs.io/en/latest/common/lightning_module.html#inference).
You can do inference with it suing `model.forward(x)`.

### Reproduce graphical materials

> TODO

### Results

> TODO
