import threading

import gym
import gym.envs.atari
import gym.wrappers
import numpy as np

class Atari(gym.Env):

    def __init__(self,
                 name,
                 frame_skip=5,
                 size=64,
                 grayscale=False,
                 terminal_on_life_loss=False,
                 sticky_actions=True,
                 noops=30):
        env = gym.envs.atari.AtariEnv(
            game=name,
            obs_type='image',
            frameskip=frame_skip,
            repeat_action_probability=0.25 if sticky_actions else 0.0
        )
        env = gym.wrappers.AtariPreprocessing(
            env, 
            noop_max=noops, 
            frame_skip=0,
            screen_size=size,
            terminal_on_life_loss=terminal_on_life_loss, 
            grayscale_obs=grayscale
        )
        self.env = env
        