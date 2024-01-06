# encoding = utf-8
import sys
import os

import gym
from gym import spaces
from gym.utils import seeding
from time import sleep
import signal

class Grid(object):
    def __init__(self,
                 x: int = None,
                 y: int = None,
                 grid_type: int = 0, #(0:空，1: 障碍/边界)
                 enter_reward: float = 0.0 # reward
                 ):
        self.x = x
        self.y = y
        self.grid_type = grid_type
        self.enter_reward = enter_reward
        self.name = f'X{self.x}-Y{self.y}'

    def __str__(self):
        return f'Grid:{{ name:{self.name}\\}}, x:{self.x}, y:{self.x}, grid_type:{self.grid_type}'
    

class GridMatrix(object):
    def __init__(self, 
                 n_width: int, # 水平方向格子数
                 n_height: int, # 垂直方向格子数
                 default_type: int = 0, # 默认类型 0 - 空
                 default_reward: float = 0, # 默认即时奖励
                 ) -> None:
        self.n_width = n_width
        self.n_height = n_height
        self.default_reward = default_reward
        self.default_type = default_type
        self.grids = None
        self.len = n_width
        self.reset()

    def reset(self):
        self.grids = []

        for y in range(self.n_height):
            for x in range(self.n_width):
                self.grids.append(Grid(x, y, self.default_type, self.default_reward))

    def get_grid(self, x, y=None) -> Grid:
        xx, yy = None, None
        if isinstance(x, int):
            xx, yy = x, y
        elif isinstance(x, tuple):
            xx, yy = x[0], y[0]

        assert(0 <= xx < self.n_width and 0 <= yy < self.n_height)

        index = yy * self.n_width + xx

        return self.grids[index]
    
    def set_reward(self, x, y, reward):
        grid = self.get_grid(x, y)
        if grid is not None:
            grid.enter_reward = reward
        else:
            raise f'grid is not exists ({x}, {y})'
    
    def set_type(self, x, y, grid_type):
        grid = self.get_grid(x, y)
        if grid is not None:
            grid.grid_type = grid_type
        else:
            raise f'grid is not exists ({x}, {y})'
        
    def  get_reward(self, x, y):
        grid = self.get_grid(x, y)
        if grid is None:
            return None
        return grid.enter_reward
    
    def get_type(self, x, y):
        grid = self.get_grid(x, y)
        if grid is None:
            return None
        return grid.grid_type
    

class GridWordEnv(gym.Env):
    metadata = {
        'render.modes':['human', 'rbg_array'], 
        'video.frames_per_second': 30
        }
    
    def __init__(self,
                 n_width: int = 5,
                 n_height: int = 5,
                 u_size = 40,
                 default_reward: float = 0,
                 default_type = 0) -> None:
        self.n_width = n_width
        self.n_height = n_height
        self.default_reward = default_reward
        self.default_type = default_type
        self.u_size = u_size
        self.screen_width = u_size * n_width # 场景宽度
        self.screen_height = u_size * n_height # xx高度

        self.grids = GridMatrix(n_width=n_width,
                                n_height=n_height,
                                default_reward=default_reward,
                                default_type=default_type)
        
        self.reward = 0 # fo rending
        self.action = None
        self.action_space = spaces.Discrete(4)
        self.observation_space = spaces.Discrete(self.n_height * self.n_width)

        self.state = None
        self.ends = [(0, 0), (4, 3)]
        self.start = (0, 2)
        self.types = [(2, 2, 1)]
        self.rewards = [(0, 0, 1), (4, 3, 5), (2, 2, -10)]

        self.refresh_setting()
        self.viewer = None
        self.seed()
        self.reset()

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]
    
    def step(self, action):
        assert self.action_space.contains(action), f'{action} ({type(action)} invalid)'
        self.action = action
        old_x, old_y = self._state_to_xy(self.state)
        new_x, new_y = old_x, old_y

        if action == 2:
            new_x -= 1
        elif action == 3:
            new_x += 1
        elif action == 0:
            new_y += 1
        elif action == 1:
            new_y -= 1
        # elif action == 4:
        #     new_x, new_y = new_x -1, new_y - 1
        # elif action == 5:
        #     new_x, new_y = new_x +1, new_y - 1
        # elif action == 6:
        #     new_x, new_y = new_x -1, new_y +1
        # elif action == 7:
        #     new_x, new_y = new_x +1, new_y + 1

        if new_x < 0:
            new_x = 0
        if new_y < 0:
            new_y = 0
        if new_x >= self.n_width:
            new_x = self.n_width - 1
        if new_y >= self.n_height:
            new_y = self.n_height - 1
        
        if self.grids.get_type(new_x, new_y) == 1:
            new_x, new_y = old_x, old_y

        self.reward = self.grids.get_reward(new_x, new_y)
        self.state = self._xy_to_state((new_x, new_y))
        done = self._is_end_state(new_x, new_y)

        info = {
            'from:': f'x:{old_x},y:{old_y}',
            'action': action,
            'dst': f'x:{new_x},y:{new_y}',
            'dst_reward': f'{self.grids.get_reward(new_x, new_y)}',
            # 'grids': self.grids
            }
        return self.state, self.reward, done, info
    
    def _state_to_xy(self, s):
        x = s % self.n_width
        y = int((s - x) / self.n_width)
        return x, y

    def _xy_to_state(self, x, y=None):
        if isinstance(x, int):
            assert (isinstance(y, int)), 'incomplete position info'
            return x + self.n_width * y
        elif isinstance(x, tuple):
            return x[0] + self.n_width * x[1]
        return -1
    
    def refresh_setting(self):
        for x, y, r in self.rewards:
            self.grids.set_reward(x, y, r)

        for x, y, t in self.types:
            self.grids.set_type(x, y, t)

    def reset(self):
        self.state = self._xy_to_state(self.start)
        return self.state
    
    def _is_end_state(self, x, y=None):
        xx, yy = -1, -1
        if y is not None:
            xx, yy = x, y
        elif isinstance(x, int):
            xx, yy = self._state_to_xy(x)
        else:
            assert(isinstance(x, tuple)), '坐标数据不完整'
            xx, yy = x[0], x[1]

        for end in self.ends:
            if xx == end[0] and yy == end[1]:
                return True
            
        return False
    
    def render(self, mode='human', close=False):
        from gym.envs.classic_control import rendering
        if close:
            if self.viewer is not None:
                self.viewer.close()
                self.viewer = None
            return 
        
        zero = (0, 0)
        u_size = self.u_size
        m = 2 #格子间的间隔

        if self.viewer is None:
            self.viewer = rendering.Viewer(self.screen_width, self.screen_height)

            for x in range(self.n_width):
                for y in range(self.n_height):
                    v = [ (x * u_size + m, y * u_size+ m), 
                        ((x + 1) * u_size - m, y * u_size + m),
                        ((x + 1) * u_size - m, (y + 1) * u_size - m),
                        (x * u_size + m, (y + 1) * u_size - m)]
                    
                    rect = rendering.FilledPolygon(v)
                    r = self.grids.get_reward(x, y) / 10
                    if r < 0:
                        rect.set_color(0.9 - r, 0.9 + r, 0.9 + r)
                    elif r > 0:
                        rect.set_color(0.3, 0.5 + r, 0.3)
                    else:
                        rect.set_color(0.9, 0.9, 0.9)

                    self.viewer.add_geom(rect)

                    # 边框
                    v_outline = [(x * u_size + m, y * u_size + m),
                                ((x + 1) * u_size - m, y * u_size + m),
                                ((x + 1) * u_size - m, (y + 1) * u_size - m),
                                (x * u_size + m, (y + 1) * u_size - m)]
                    outline = rendering.make_polygon(v_outline, False)
                    outline.set_linewidth(3)

                    if self._is_end_state(x, y):
                        outline.set_color(0.9, 0.9, 0) # 金黄色边框
                        self.viewer.add_geom(outline)
                    if self.start[0] == x and self.start[1] == y:
                        outline.set_color(0.5, 0.5, 0.8)
                        self.viewer.add_geom(outline)
                    if self.grids.get_type(x, y) == 1:
                        rect.set_color(0.3, 0.3, 0.3)
                    else:
                        pass

            # 绘制个体
            self.agent = rendering.make_circle(u_size / 4, 30, True)
            self.agent.set_color(1.0, 1.0, 0.0)
            self.viewer.add_geom(self.agent)
            self.agent_trans = rendering.Transform()
            self.agent.add_attr(self.agent_trans)

        x, y  = self._state_to_xy(self.state)
        self.agent_trans.set_translation((x + 0.5) * u_size, (y + 0.5) * u_size)
        # for i in range(0, self.n_width):
        #     for j in range(0, self.n_height):
        #         print(f'{i}:{j}:{self.grids.get_type(i, j)}')
        return self.viewer.render(return_rgb_array= mode == 'rgb_array')


def CtrlCHandler(signum, frame):
    env.close()
    print('User interrupt')
    sys.exit(0)


if __name__ == '__main__':
    env = GridWordEnv()
    env.refresh_setting()
    env.seed(1)

    print('env initaled')
    signal.signal(signal.SIGINT, CtrlCHandler)
    episode_num = 100
    for e in range(episode_num):
        print(f'epis {e}')
        env.reset()
        while True:
            action = env.action_space.sample()
            sleep(0.5)
            _, _, done, info = env.step(action)
            # env.render()
            print(f'info:{info}')
            if done:
                break
    env.close()
    sys.exit(0)