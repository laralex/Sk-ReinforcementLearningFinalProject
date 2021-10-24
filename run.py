from src.critic import Critic
from src.actor import Actor
from src.code_level_optim import CodeLevelOptimizations, StdNormalizer

import src.utility as utility
import gym
import tqdm

EPISODE_TIME_LIMIT = 1*2048*5

def env_loop(env, timesteps, config, do_render=False):
   timesteps = tqdm.tqdm(timesteps)

   rewards_normalizer = StdNormalizer()
   gamma = config['discount']
   optimizations_context = config['code_level_optim']

   n_episodes = max(config['critic']['n_epochs'], config['actor']['n_epochs'])

   for episode in range(n_episodes):
      for timestep_idx in timesteps:
         timesteps.set_description(f"Timestep {timestep_idx}")

         obs, reward, done, info = env.step(env.action_space.sample())

         obs = CodeLevelOptimizations.normalize_state(optimizations_context, obs)

         rewards_normalizer.add_raw_reward(reward, gamma)
         reward, returns = CodeLevelOptimizations.normalize_rewards(
            optimizations_context, rewards_normalizer, gamma, reward, None) # todo returns

         if timestep_idx % config['timesteps_per_iteration'] == 0:
            # update critic-actor
            # loss of critic
            # optimizer backward
            # loss of actor
            # optimizer backward
            pass

         if done:
            break

         if do_render:
            env.render()
      # schedulers update

def main():
   args = utility.parse_args()
   config = utility.parse_config(args.config_path)

   print('== CONFIG OF THE EXPERIMENT ==')
   utility.pretty_print(config)

   env = gym.make(config['gym_env'])
   env._max_episode_steps = EPISODE_TIME_LIMIT
   env.reset()
   env_loop(env, range(0, EPISODE_TIME_LIMIT, 5), config, do_render=args.render)
   env.close()

if __name__ == "__main__":
   main()