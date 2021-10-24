import yaml
import sys
import argparse

def parse_args():
   parser = argparse.ArgumentParser(description='Process some integers.')
   parser.add_argument('config_path', type=str,
                     help='Path to `./configs/*.yaml` file with environment configuration')
   parser.add_argument('--render', action='store_true',
                     help='Render 3D episodes in a graphical window')

   return parser.parse_args()

def parse_config(filepath):
   with open(filepath, "r") as stream:
      config = yaml.safe_load(stream)
   level_1_keys = ["gym_env", "critic", "trpo", "actor", "ppo", "code_level_opt"]
   assert all(level_1 in config.keys() for level_1 in level_1_keys)

   # set defaults
   config.get('timesteps_per_iteration', 2048)
   config.get('discount', 0.99)
   config.get('gae_discount', 0.95)
   config['critic'].get('hidden_layers', [42])
   config['critic'].get('lr', 0.0005)
   config['critic'].get('n_epochs', 10)
   config['actor'].get('hidden_layers', [42])
   config['actor'].get('lr', 0.0005)
   config['actor'].get('n_epochs', 10)
   config['trpo'].get('kl_constraint', None)
   config['trpo'].get('fisher_estim_fraction', None)
   config['trpo'].get('conjugate_grad_steps', None)
   config['trpo'].get('conjugate_grad_damping', None)
   config['trpo'].get('backtracking_steps', None)
   config['ppo'].get('clipping_epsilon', None)
   config['code_level_opt'].get('critic_loss_clpping', False)
   config['code_level_opt'].get('entropy_coefficient', None)
   config['code_level_opt'].get('reward_clipping', None)
   config['code_level_opt'].get('state_clipping', None)
   config['code_level_opt'].get('gradient_clipping_l2', None)
   config['code_level_opt'].get('activation_func', None)
   config['code_level_opt'].get('state_normalization', False)
   config['code_level_opt'].get('returns_normalization', False)
   config['code_level_opt'].get('rewards_normalization', False)
   config['code_level_opt'].get('actor_annealing_class', None)
   config['code_level_opt'].get('actor_annealing_kwargs', None)
   config['code_level_opt'].get('critic_annealing_class', None)
   config['code_level_opt'].get('critic_annealing_kwargs', None)
   return config

def pretty_print(dictionary):
   print(yaml.dump(dictionary, default_flow_style=False))