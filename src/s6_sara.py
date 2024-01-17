import numpy as np
from collections import defaultdict
from  roomba_env import GridWordEnv

class Sara(GridWordEnv):
    def __init__(self, n_width: int = 5, 
                 n_height: int = 5, 
                 u_size=40, 
                 default_reward: float = 0, 
                 default_type=0,
                 action_len = 4, params = {}) -> None:
        
        super().__init__(n_width, n_height, u_size, 
                         default_reward, default_type)
        self.Q = {}
        self.action_len = action_len
        self.init_valid_actions()

    def init_valid_actions(self):

        self.valid_actions = {}

        for s in range(0, self.observation_space.n):
            self.valid_actions[s] = []
            for a in range(0, self.action_len):
                nxs = self.step_from_state(s, a, True)
                if nxs != s:
                    self.valid_actions[s].append(a)
            
        
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
    
    def _is_state_in_q(self, state):
        return self.Q.get(state) is not None
    
    def _get_q_value(self, s, a):
        return self.Q[s][a]

    def _set_q_value(self, s, a, new_q):
        self.Q[s][a] = new_q
    
    def init_value(self, s, randomized=False):
        if not self._is_state_in_q(s):
            self.Q[s] = {}
            for a in self.valid_actions[s]:
                self.Q[s][a] = np.random().random() / 10 if randomized else 0.0
                
    def get_decrease_weight(self, weight, episode_index, max_episode_num):
        rweight = ((float(episode_index) - float(max_episode_num)) / float(max_episode_num)) * max_episode_num
        return rweight
            
    def sara(self, alpha, gamma, epsilon, max_episode_num):

        num_episode = 0
        for state in range(self.observation_space.n):
            self.init_value(state)

        while num_episode < max_episode_num:
            delta = 0.0
            state = self.reset()
            alpha = self.get_decrease_weight(alpha, num_episode, max_episode_num)
            epsilon = self.get_decrease_weight(epsilon, num_episode, max_episode_num)
            
            probs = self.get_policy(self.valid_actions, state, epsilon)
            action = np.random.choice(np.arange(len(probs)), p = probs) 
            
            done = False
            while True:
                # probs = self.get_policy(self.valid_actions, state, epsilon)
                # action = np.random.choice(np.arange(len(probs)), p = probs) 
                next_state, reward, done, _ = self.step(action)
                if done:
                    break
                next_probs = self.get_policy(self.valid_actions, next_state, epsilon)
                next_action = np.random.choice(np.arange(len(probs)), p = next_probs) 
                qa_value = self.Q[state][action] + alpha * (reward + gamma * self.Q[next_state][next_action] - self.Q[state][action])
                # qa_value = self.Q[state][action] + (1 / returns_count[sa_pair])*(G - self.Q[state][action])
                oqa = self._get_q_value(state, action)
                self._set_q_value(state, action, qa_value) # TO DO check
                delta = max(delta, np.abs(qa_value - oqa))
            print(f'iteration {num_episode} delta is {delta}')
            
        return self.Q


    def human_view(self, vec):
        t_sque = [[n * self.n_width + e for e in range(0, self.n_width)] for n in range(self.n_height - 1, -1, -1)]
        for line_lst in t_sque:
            line_lst = [np.round(vec[item], decimals=3) for item in line_lst]
            line = ' '.join(list(map(str, line_lst)))
            print(line)

if __name__ == '__main__':
    ss = Sara()
    alpha = 0.05
    gamma = 0.8
    epsilon = 0.5
    max_episode_num=20000
    ss.sara(alpha, gamma, epsilon, max_episode_num)
