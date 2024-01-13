import numpy as np
from collections import defaultdict
from  roomba_env import GridWordEnv

class FirstVisitGreedyMC(GridWordEnv):
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

    def init_valid_actions(self, s: int):

        self.valid_actions = {}

        for s in range(0, len(self.observation_space)):
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
        best_action = max(self.Q[state], key=self.Q[state].get)
        A[best_action] += 1.0 - epsilon
        return A

                
    def get_ext_action(self, action):
        if action == 0:
            return 1
        elif action == 1:
            return 0
        elif action == 2:
            return 3
        elif action == 3:
            return 2
        else:
            action = -1

        assert self.action_space.contains(action), f'{action} is out of space'
        
        return -1
    
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
            
    def mc_control(self, gamma, max_episode_num):
        returns_sum = defaultdict(float)
        returns_count = defaultdict(float)
        target_policy = self.get_policy
        num_episode = 0
        for state in range(len(self.observation_space)):
            self.init_value(state, )



    def human_view(self, vec):
        t_sque = [[n * self.n_width + e for e in range(0, self.n_width)] for n in range(self.n_height - 1, -1, -1)]
        for line_lst in t_sque:
            line_lst = [np.round(vec[item], decimals=3) for item in line_lst]
            line = ' '.join(list(map(str, line_lst)))
            print(line)

if __name__ == '__main__':
    # ss = StateValueRVSimulate()
    # ss.init_policy()
    # ss.eval_policy()
    pass