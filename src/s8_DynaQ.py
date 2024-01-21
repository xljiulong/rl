import numpy as np
import random
from collections import defaultdict
from  roomba_env import GridWordEnv

class DynaQ(GridWordEnv):
    def __init__(self,
                 n_width: int = 5,
                 n_height: int = 5,
                 u_size = 40,
                 default_reward: float = 0,
                 default_type = 0) -> None:
        super().__init__(n_width, n_height, u_size, 
                         default_reward, default_type)
        self.episode = 1
        self.Q = {}
        self.model = {}
        self.actions = [0, 1, 2, 3]
        self.position =self.start
        
    def update_model(self, pos, act, reward, next_state):
        key = (pos, act)
        value = (reward, next_state)
        self.model[key] = value

    def get_policy(self, action_lst, state, epsilon):
        # pi = [[] for s in range(self.n_height * self.n_width)]

        A = np.zeros(self.action_len, dtype=float)
        valid_nA = len(action_lst[state])
        for action in action_lst[state]:
            A[action] = epsilon / valid_nA
        best_action_id = max(self.Q[state], key=self.Q[state].get)
        best_actions_ids = [id for id in self.Q[state].keys() if self.Q[state][best_action_id] == self.Q[state][id]]
        A = [A[id] + (1.0 - epsilon) if id in best_actions_ids else A[id]  for id in range(0, len(A)) ]
        a_sum = sum(A)
        A = A / a_sum
        # A = [A[id] + (1.0 - epsilon) for id in range(0, len(A)) if id in best_actions_ids else A[id]]
        # A[best_action] += 1.0 - epsilon
        return A

    def get_maxa_from_state(self, state):
        max_q = -1000
        ret_act = -1
        for act in self.actions:
            a_q = self.get_q(state, act)
            if a_q >= max_q:
                ret_act = act

        return ret_act
    
    def get_decrease_weight(self, weight, episode_index, max_episode_num):
        rweight = ((float(max_episode_num) - float(episode_index)) / float(max_episode_num)) * weight
        return rweight

    def get_random_action_from_state(self, state, num_episode, max_episode_num):
        alpha = self.get_decrease_weight(alpha, num_episode, max_episode_num)
        epsilon = self.get_decrease_weight(epsilon, num_episode, max_episode_num)
        
        probs = self.get_policy(self.valid_actions, state, epsilon)
        action = np.random.choice(np.arange(len(probs)), p = probs)
        return action

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
                
    def study(self, n):
        for i in range(300):
            done = False
            self.reset()
            while(done == False):
                state = self._state_to_xy(self.state)
                action = self.get_random_action_from_state(state)
                nxt_state, reward, done, info = self.step(action)
                nxt_state = self._state_to_xy(nxt_state)

                self.update_q(state, action, nxt_state, reward)
                self.update_model(state, action, reward, nxt_state)
                self.q_plan(n)
    
if __name__ == '__main__':
    

    n_width = 5
    n_height = 5
    default_reward = 0
    da = DynaQ(n_width, n_height, default_reward=default_reward)

    da.types = [(2, 2, 1)]
    da.rewards = [(0, 0, 1), (4, 3, 1)]  # 奖赏值设定
    da.start = (0, 4)
    da.ends = [(0, 0), (4, 3)]
    da.refresh_setting()
    da.init_valid_actions()

    da.study(5)
