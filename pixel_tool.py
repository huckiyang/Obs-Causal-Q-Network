import numpy as np
import os, math, pandas

os.environ.setdefault('PATH', '')
from collections import deque
import gym
from gym import spaces

# for those who installed ROS on local env
import sys
from cartpole_pixel import RenderThread
import cv2

cv2.ocl.setUseOpenCL(False)

"""
Wrapper for Cartpole
This is to change the reward at the terminal state because originally it is set as 1.0
check here: https://github.com/openai/gym/blob/master/gym/envs/classic_control/cartpole.py
"""


class CartPole_Pixel(gym.Wrapper):
    """
    Wrapper for getting raw pixel in cartpole env
    observation: 400x400x1 => (Width, Height, Colour-chennel)
    we dispose 100pxl from each side of width to make the frame divisible(Square) in CNN
    """

    def __init__(self, env):
        self.width = 400
        self.height = 400

        gym.Wrapper.__init__(self, env)
        self.env = env.unwrapped
        # self.env.seed(123)  # fix the randomness for reproducibility purpose

        """
        start new thread to deal with getting raw image
        """

        self.renderer = RenderThread(env)
        self.renderer.start()

    def _pre_process(self, frame):
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        frame = cv2.resize(frame, (self.width, self.height), interpolation=cv2.INTER_AREA)
        frame = np.expand_dims(frame, -1)
        return frame

    def step(self, ac):
        _, reward, done, info = self.env.step(ac)
        self.renderer.begin_render()  # move screen one step
        observation = self._pre_process(self.renderer.get_screen())

        if done:
            reward = -1.0  # reward at a terminal state
        return observation, reward, done, info

    def reset(self, **kwargs):
        self.env.reset()
        self.renderer.begin_render()  # move screen one step
        return self._pre_process(self.renderer.get_screen())  # overwrite observation by raw image pixels of screen

    def close(self):
        self.renderer.stop()  # terminate the threads
        self.renderer.join()  # collect the dead threads and notice all threads are safely terminated
        if self.env:
            return self.env.close()