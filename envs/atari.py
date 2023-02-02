import gym
import gym.wrappers
import numpy as np


class Atari(gym.Env):

    def __init__(self,
                 name,
                 frame_skip=4,
                 size=64,
                 grayscale=False,
                 terminal_on_life_loss=False,
                 sticky_actions=False,
                 noops=30):

        env = gym.make(
            name, 
            obs_type='rgb',
            render_mode='rgb_array',
            frameskip=frame_skip,
            full_action_space=False
        )
        env = gym.wrappers.AtariPreprocessing(
            env, 
            noop_max=noops, 
            frame_skip=1,
            screen_size=size,
            terminal_on_life_loss=terminal_on_life_loss, 
            grayscale_obs=grayscale
        )
        self.env = OneHotAction(env)
        self.grayscale = grayscale
        c,h,w = self.reset()['image'].shape
        self.observation_space = gym.spaces.MultiBinary((c,h,w))

    @property
    def action_space(self):
        return self.env.action_space

    def reset(self):
        image, info = self.env.reset()
        if self.grayscale:
            image = image[..., None]
        obs = {"image": image.transpose(2, 0, 1)}
        return obs

    def step(self, action):
        image, reward, done, truncated, info = self.env.step(action)
        done = truncated or done
        if self.grayscale:
            image = image[..., None]
        obs = {"image": image.transpose(2, 0, 1)}
        return obs, reward, done, info

    def render(self, mode):
        return self.env.render(mode)
        
        
class OneHotAction(gym.Wrapper):
    def __init__(self, env):
        assert isinstance(env.action_space, gym.spaces.Discrete), "This wrapper only works with discrete action space"
        shape = (env.action_space.n,)
        env.action_space = gym.spaces.Box(low=0, high=1, shape=shape, dtype=np.float32)
        env.action_space.sample = self._sample_action
        super(OneHotAction, self).__init__(env)
    
    def step(self, action):
        index = np.argmax(action).astype(int)
        reference = np.zeros_like(action)
        reference[index] = 1
        return self.env.step(index)

    def reset(self):
        return self.env.reset()
    
    def _sample_action(self):
        actions = self.env.action_space.shape[0]
        index = np.random.randint(0, actions)
        reference = np.zeros(actions, dtype=np.float32)
        reference[index] = 1.0
        return reference