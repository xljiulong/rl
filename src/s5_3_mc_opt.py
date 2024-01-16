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
        for state in range(self.observation_space.n):
            self.init_value(state)

        while num_episode < max_episode_num:
            episode = []
            state = self.reset()
            delta = 0

            while True:
                probs = self.get_policy(self.valid_actions, state, self.get_epsilon_by_epsiode(num_episode))
                action = np.random.choice(np.arange(len(probs)), p = probs)
                next_state, reward, done, _ = self.step(action)
                episode.append((state, action, reward))
                if done:
                    break
                state = next_state

            num_episode += 1
            sa_in_episode = set([(x[0], x[1]) for x in episode])

            first_occurence_idx = next(i for i, x in enumerate(episode) 
                                       if x[0] == state and x[1] == action)
            
            # unique_episode = sorted(episode, key=episode.index)
            unique_episode = list(reversed(episode))
            G = 0
            for state, action, reward in unique_episode:  # TO DO  CHECK
                sa_pair = (state, action)

                # G = sum(x[2] * (gamma ** i) for 
                #         i, x in enumerate(episode[first_occurence_idx:])) # TO DO  CHECK
                G = gamma * G + reward

                returns_sum[sa_pair] += G
                returns_count[sa_pair] += 1.0
                qa_value = self.Q[state][action] + (1 / returns_count[sa_pair])*(G - self.Q[state][action])
                oqa = self._get_q_value(state, action)
                self._set_q_value(state, action, qa_value) # TO DO check
                delta = max(delta, np.abs(qa_value - oqa))
            print(f'iteration {num_episode} delta is {delta}')
        return self.Q



    def get_epsilon_by_epsiode(self, epsiode): 
        epsilon_start = 0.5
        epsilon_final = 0
        epsilon_episodes = 20000
        epsilon_by_epsiode =  epsilon_start - (epsilon_start - epsilon_final) * min(epsiode,  epsilon_episodes) / epsilon_episodes
        return epsilon_by_epsiode

    def human_view(self, vec):
        t_sque = [[n * self.n_width + e for e in range(0, self.n_width)] for n in range(self.n_height - 1, -1, -1)]
        for line_lst in t_sque:
            line_lst = [np.round(vec[item], decimals=3) for item in line_lst]
            line = ' '.join(list(map(str, line_lst)))
            print(line)

if __name__ == '__main__':
    ss = FirstVisitGreedyMC()
    gamma = 0.8
    max_episode_num=20000
    ss.mc_control(gamma, max_episode_num)
