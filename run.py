from src.critic import Critic
from src.actor import Actor
from src.code_level_optim import CodeLevelOptimizations, StdNormalizer

import src.utility as utility
import gym
import tqdm
import itertools
import torch
import torch.nn as nn

EPISODE_TIME_LIMIT = 1*2048*5

def env_loop(env: gym.Env, timesteps, config, do_render=False):
   timesteps = tqdm.tqdm(timesteps)

   rewards_normalizer = StdNormalizer()
   gamma = config['discount']
   code_level_context = config['code_level_optim']

   n_episodes = max(config['critic']['n_epochs'], config['actor']['n_epochs'])

   action_size = env.action_space.sample()
   state_size = env.observation_space.sample()

   critic = Critic(config['critic'], code_level_context, input_size=(action_size + state_size))
   critic_optimizer = torch.optim.AdamW(critic.parameters(),
      lr = config['critic']['lr'], betas=(0.9, 0.999), weight_decay=1e-3)
   critic_scheduler = CodeLevelOptimizations.make_critic_lr_annealing(
      code_level_context, critic_optimizer)

   actor = Actor(config['actor'], code_level_context, input_size=state_size)
   actor_optimizer = torch.optim.AdamW(actor.parameters(),
      lr = config['actor']['lr'], betas=(0.9, 0.999), weight_decay=1e-3)
   actor_scheduler = CodeLevelOptimizations.make_actor_lr_annealing(
      code_level_context, critic_optimizer)

   for episode in range(n_episodes):
      for timestep_idx in range(0, EPISODE_TIME_LIMIT, 5):
         timesteps.set_description(f"Timestep {timestep_idx}")

         state, reward, done, info = env.step(env.action_space.sample())

         state = CodeLevelOptimizations.clip_state(code_level_context, state)
         state = CodeLevelOptimizations.normalize_state(code_level_context, state)

         rewards_normalizer.add_raw_reward(reward, gamma)
         reward, returns = CodeLevelOptimizations.normalize_rewards(
            code_level_context, rewards_normalizer, gamma, reward, 0.0) # todo returns
         reward = CodeLevelOptimizations.clip_reward(reward)

         # optimize actor-critic networks for the latest portion of episode
         if timestep_idx % config['timesteps_per_iteration'] == 0:
            critic_optimizer.zero_grad()
            actor_optimizer.zero_grad()

            # compute critic loss
            if episode < config['critic']['n_epochs']:
               critic_loss = 0.0
               critic_loss.backward()

            # compute actor loss
            if episode < config['actor']['n_epochs']:
               actor_loss = 0.0
               actor_loss.backward()

            global_parameters = itertools.chain(actor.parameters(), critic.parameters())
            CodeLevelOptimizations.clip_gradient(code_level_context, global_parameters)

            critic_optimizer.step()
            actor_optimizer.step()

         if done:
            break

         if do_render:
            env.render()
      
      # lr annealing after episode (aka epoch)
      CodeLevelOptimizations.anneal_learning_rate(critic_scheduler)
      CodeLevelOptimizations.anneal_learning_rate(actor_scheduler)

def main():
   args = utility.parse_args()
   config = utility.parse_config(args.config_path)

   print('== CONFIG OF THE EXPERIMENT ==')
   utility.pretty_print(config)

   env = gym.make(config['gym_env'])
   env._max_episode_steps = EPISODE_TIME_LIMIT
   env.reset()
   env_loop(env, config, do_render=args.render)
   env.close()

if __name__ == "__main__":
   main()