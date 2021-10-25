from .code_level_optim import CodeLevelOptimizations

import numpy as np
import torch
import torch.nn as nn

def get_policy_entropy(n_samples, range_min, range_max):
   return 0.0

class Actor(nn.Module):
   def __init__(self, actor_config, code_level_config, input_size, output_size):
      super().__init__()
      self.input_size = input_size
      hidden_layers = actor_config['hidden_layers']
      assert isinstance(hidden_layers, list)

      prev_size = input_size
      self.hidden_layers = []

      self.code_level_context = code_level_config

      for layer_size in hidden_layers:
         layer = nn.Linear(prev_size, layer_size)
         CodeLevelOptimizations.initialize_layer(
            self.code_level_context, layer)
         self.hidden_layers.append(layer)
         prev_size = layer_size

      self.final_layer = nn.Linear(prev_size, output_size)
      CodeLevelOptimizations.initialize_layer(
         self.code_level_context, layer, orthogonal_gain=1.0)

   def forward(self, x):
      for layer in self.hidden_layers:
         x = layer(x)
         x = CodeLevelOptimizations.activation_func(
            self.code_level_context, x)

      return self.final_layer(x)