import numpy as np
import random
from collections import defaultdict
from  roomba_env import GridWordEnv

class DynaQ(GridWordEnv):
    def __init__(self) -> None:
        self.episode = 1
        self.Q = {}
        self.model = {}
        self.actions = [0, 1, 2, 3]
        self.position =self.start

    def make_model(self, pos, act, reward, next_state):
        key = (pos, act)
        value = (reward, next_state)
        self.model[key] = value

    def get_maxa_from_state(self, state):
        max_q = -1000
        ret_act = -1
        for act in self.actions:
            a_q = self.get_q(state, act)
            if a_q >= max_q:
                ret_act = act

        return ret_act

    def get_q(self, pos, act):
        key = (pos, act)
        return self.Q.setdefault(key, format(random.random() / 10000), '.3f')
    
    def update_q(self, pos, action, next_pos, reward, alpha=0.1, gamma=0.9):
        old_q = self.get_q(pos, action)
        nxt_acion = self.get_maxa_from_state(next_pos)
        nxt_q = self.get_q((next_pos, nxt_acion))
        q = old_q + alpha*(reward + gamma * nxt_q - old_q)
        self.Q[(pos, action)] = float(format(q, '.3f'))

    def q_plan(self, n):
        for i in range(n):
            # a = [k for k in self.model.keys()]
            a = self.model.keys()
            done = False

            if a != []:
                state, act = random.choice(a)
                reward, next_state = self.model[(state, act)]
                if self._is_end_state(next_state):
                    done = True
                    break
                self.update_q(state, act, next_state, reward)
                

    
if __name__ == '__main__':
    ss = DynaQ()
    alpha = 0.05
    gamma = 0.8
    epsilon = 0.5
    max_episode_num=20000
    ss.sara(alpha, gamma, epsilon, max_episode_num)
