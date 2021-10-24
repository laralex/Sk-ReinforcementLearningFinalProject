from typing_extensions import ParamSpecArgs

from tqdm.utils import _environ_cols_wrapper
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class StdNormalizer:
   """
   Implements rewards normalization, by computing standard deviation
   of a history of cumulative rewards.
   See Appending A.2 of the reference paper
   """
   def __init__(self):
      self.history  = [0.0]
      self.history_sum     = 0.0
      self.history_sum_sqr = 0.0

   def add_raw_reward(self, current_reward, gamma):
      self.history.append(gamma*self.history[-1] + current_reward)
      self.history_sum     += self.history[-1]
      self.history_sum_sqr += self.history[-1]**2

   def __normalize(self, raw_value):
      # variance(X) = [ (\sum x^2) - (\sum x)^2/n ] / (n-1)
      N = len(self.history)
      var = ( self.history_sum_sqr - self.history_sum**2 / N ) / (N - 1)
      std = np.sqrt(var)
      return raw_value / (std + 1e-8)


class CodeLevelOptimizations:
   """
   Provides functional style code-level optimizations,
   as described in the reference paper.
   Pass `context` object to these functions, and if it prescribes to use
   a certain optimization, it will apply it.
   If optimization is not prescribed, the corresponding functions
   act as identity / noop functions.
   """

   # <Optimization #1>
   """
   Returns loss of value network
   unclipped mode: L = (V_theta - V_target)^2
   clipped mode: L = max[
      (V_theta - V_target)^2,
      (clip(V_theta, V_prev_theta-eps, V_prev_theta+eps) - V_target)^2 ]
   Where `eps` equals to PPO clipping constant
   """
   def get_critic_loss(context, current_values, old_values, advantages, ppo_epsilon):
      target_values = old_values + advantages
      unclipped_loss = (values - target_values).pow(2)
      if context['critic_loss_clpping']:
         clipped_values = torch.clamp(current_values,
            min=old_values - ppo_epsilon, max=old_values + ppo_epsilon)
         clipped_loss = (clipped_values - target_values).pow(2)
         return max(unclipped_loss, clipped_loss)
      else:
         return unclipped_loss

   # <Optimization #2>
   def normalize_rewards(context: dict, normalizer: StdNormalizer, gamma, reward, returns):
      """
      Takes new raw reward, updates inner statistics and applies normalization
      to the given reward and/or return, depending on flags
      context['returns_normalization'] and context['rewards_normalization']
      """
      normalizer.add_raw_reward(reward, gamma)
      if context['returns_normalization']:
         returns = normalizer.__normalize(returns)
      if context['rewards_normalization']:
         reward = normalizer.__normalize(reward)
      return reward, returns

   # <Optimization #3>
   def initialize_layer(context: dict, layer: nn.Module, orthogonal_gain=None):
      """
      Initializes the given layer
      If context['layers_initialization'] is "xavier", use Xavier initialization
      If context['layers_initialization'] is "orthogonal", use orthogonal initialization
      with the given gain
      The gain can be determined from context['actionvation_func'], if orthogonal_gain=None
      """
      if len(layer.data.shape) < 2:
         layer.data.zero_()
         return

      if context['layers_initialization'] == "orthogonal":
         if orthogonal_gain is None:
            if context['activation_func'] == "relu":
               orthogonal_gain = np.sqrt(2)
            elif context['activation_func'] == "tanh":
               orthogonal_gain = 1.0
            else:
               raise NotImplementedError("Unsupported activation")
         nn.init.orthogonal_(layer.data, gain=orthogonal_gain)
      elif context['layers_initialization'] == "xavier":
         nn.init.xavier_uniform_(layer.data)
      else:
         raise NotImplementedError("Unsupported initialization")


   # <Optimization #4>
   def __make_lr_annealing(class_name, optimizer, **kwargs):
      """
      Initialization part of learning rate annealing optimization
      Returns a scheduler or None
      """
      if class_name is None:
         return None

      SchedulerClass = getattr(torch.optim.lr_scheduler, class_name)
      return SchedulerClass(optimizer, **kwargs)

   def make_actor_lr_annealing(context: dict, optimizer):
      return __make_lr_annealing(context['actor_annealing_class'],
                                 **context['actor_annealing_kwargs'])

   def make_critic_lr_annealing(context: dict, optimizer):
      return __make_lr_annealing(context['critic_annealing_class'],
                                 **context['critic_annealing_kwargs'])

   def anneal_learning_rate(context, scheduler):
      """ Update the given sheduler. Noop if the scheduler is None """
      if scheduler is None:
         return
      scheduler.step()

   # <Optimization #5>
   def clip_reward(context: dict, reward):
      """
      Clip rewards to [min, max] range
      before the rest of the system will see them
      * Enabled if `context.reward_clipping` is a tuple of floats
      """
      if context['reward_clipping'] is None:
         return reward
      min_val, max_val = context['reward_clipping']
      return torch.clamp(reward, min_val, max_val)

   # <Optimization #6>
   def normalize_state(context: dict, state):
      """
      Normalize raw observations to mean-zero, variance-one,
      before the rest of the system will see them
      * Enabled if `context['state_clipping']` is a tuple of floats
      """
      if not context['state_normalization']:
         return state
      return (state-state.mean())/(state.std() + 1e-8)

   # <Optimization #7>
   def clip_state(context: dict, state):
      """
      Clip observations to [min, max] range
      before the rest of the system will see them
      * Enabled if `context['state_clipping']` is a tuple of floats
      """
      if context['state_clipping'] is None:
         return state
      min_val, max_val = context['state_clipping']
      return torch.clamp(state, min_val, max_val)

   # <Optimization #8>
   def activation_func(context: dict, activations):
      if context['activation_func'] == "relu":
         return F.relu(activations)
      elif context['activation_func'] == "tanh":
         return F.tanh(activations)
      elif context['activation_func'] is not None:
         raise NotImplementedError("Unsupported activation")
      else:
         return activations

   # <Optimization #9>
   def clip_gradient(context: dict, parameters):
      """
      Clip gradient magnitude to specified L2-norm value
      * Enabled if `context.gradient_clipping_l2` is float
      """
      if context['gradient_clipping_l2'] is None or context['gradient_clipping_l2'] < 0.0:
         return
      nn.utils.clip_grad_norm(
         parameters, context['gradient_clipping_l2'], norm_type=2.0)
