# Final project for 'Reinforcement Learning' course, 2021, Skoltech
Reimplementation of the original paper 

> [Implementation Matters in Deep Policy Gradients: A Case Study on PPO and TRPO](https://arxiv.org/abs/2005.12729)

> [GitHub repository of the paper](https://github.com/MadryLab/implementation-matters)

Team members:
* Alexey Larionov (IST MSc-2)
* Ilya Barskiy (IST MSc-2)

### Repository overview
* 👉 [`run.py`](run.py) - the loop of training for the given config
* [`configs/`](configs/) - YAML files with parameters of all experiments
* 👉[`src/code_level_optim.py`](src/code_level_optim.py) - all the code for code-level optimizations
  as described in the referenced paper (all implemented in functional style to
  use the same training code with and w/o optimizations)
* [`src/actor.py`](src/actor.py) - learnable policy Neural Network (MLP)
* [`src/critic.py`](src/critic.py) - learnable value function Neural Network
  (MLP)
* [`src/utility.py`](src/utility.py) - command line arguments parsing,
  experiment's config parsing
* [`materials/`](materials/) - images, the presentation, the original paper file

### Requirements
* See [`requirements.txt`](requirements.txt) with dependencies of this project.
Note that it implies, that you have MuJoCo properly installed for your
operating system. For example, use [this
  guide](https://medium.com/@sayanmndl21/install-openai-gym-with-box2d-and-mujoco-in-windows-10-e25ee9b5c1d5)
  for Windows.
* [`PyTorch`](http://pytorch.org/) is used for neural networks implementation.
* You may also want to install [OpenAI
  Baselines](https://github.com/openai/baselines) to check out performance of
  standard implementations of TRPO and PPO

### Reproduce
The usage is meant to be as simple as

```
python run.py --render configs/{config_name}.yaml
```

Where `--render` flag is optional, and path to a YAML config is obligatory.

This shall start episodic optimization of the actor-critic in the given
environment with the specified configuration.

### Results

Unfortunately due to struggles with setting up MuJoCo environments on Windows
and because of external overload, we didn't manage to reimplement TRPO in time as
described in [its paper](https://arxiv.org/abs/1502.05477) (and we didn't want
to copy-paste implementation from the [Implementation matters
repository](https://github.com/MadryLab/implementation-matters)).

Thus we don't have our own results to present. Apologies for that! 

But we want to reference the paper's results (not ours), that in fact code-level
optimizations are what majorily contributes to PPO performance. In the table you
can see that difference between PPO-M (PPO with code-level optimizations
removed) and true TRPO is very small, while default PPO is by far better.
Moreover, in TRPO+ (TRPO with code-level optimizations added) performs similarly
(if not better) than default PPO.

![image here](https://github.com/laralex/Sk-reinforcement-learning/blob/main/materials/paper-result.PNG)
