import numpy as np

from s4_1_state_value_dv_opt import StateValueDVSimulate

class StateValueRVSimulate(StateValueDVSimulate):
    def __init__(self, n_width: int = 5, 
                 n_height: int = 5, 
                 u_size=40, 
                 default_reward: float = 0, 
                 default_type=0,
                 action_len = 4) -> None:
        
        super().__init__(n_width, n_height, u_size, 
                         default_reward, default_type)
            
        
    def init_policy(self):
        # pi = [[] for s in range(self.n_height * self.n_width)]

        for s in range(self.n_width * self.n_height):
            for a in range(self.action_len):
                if s == self.step_from_state(s, a, skip_type=True):
                    continue
                successor_num = len(self.get_successors(s))
                self.pi[s][a] = 1.0/successor_num
                
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
    

    def eval_policy(self, theta = 0.001, eval_round=100):
        V = np.zeros(self.n_width * self.n_height)
        gamma = 0.8
        t_sque = [[n * self.n_width + e for e in range(0, self.n_width)] for n in range(self.n_height - 1, -1, -1)]
        sque = []
        for lt in t_sque:
            sque.extend(lt)
            
        for iter in range(0, eval_round):
            k = -1
            delta = 0
            for state in range(self.n_width * self.n_height):
                # v = V[state]
                v = 0
                if self.grids.get_type(state) == 1:
                    print(f'state {state} will not get evaled')
                    continue

                if self.grids.get_reward(state) > 0:
                    continue

                # 迭代
                for action in range(0, self.action_len):
                    cur_reward = self.get_reward(state)

                    nxt_sate = self.step_from_state(state, action, skip_type=True)
                    nxt_reward = self.get_reward(nxt_sate)

                    ext_action = self.get_ext_action(action)
                    ext_state = self.step_from_state(state, ext_action)
                    ext_reward = self.get_reward(ext_state)

                    if self.grids.get_type(nxt_sate) == 1:
                        s2v =  V[state]
                    else:
                        s2v = V[nxt_sate]

                    if self.grids.get_type(ext_state) == 1:
                        s3v = V[state]
                    else:
                        s3v = V[nxt_sate]
                        # v += self.pi[state][action] * (reward + gamma * V[state])
                    v += self.pi[state][action] *( 0.8 * (nxt_reward + gamma * V[state]) + 0.15 * (cur_reward + gamma * s2v) + 0.05 * (ext_reward + gamma * s3v))

                    
                delta = max(delta, np.abs(v - V[state]))
                V[state] = v
            print(f'iter {iter} delta is {delta}')
            # value = np.array(V).reshape(self.n_width, self.n_height)
            # print(f'value is::')
            # self.human_view(V)
            

            if delta <= theta:
                print(f'delta is good enough {delta}')
                exit(0)

    def human_view(self, vec):
        t_sque = [[n * self.n_width + e for e in range(0, self.n_width)] for n in range(self.n_height - 1, -1, -1)]
        for line_lst in t_sque:
            line_lst = [np.round(vec[item], decimals=3) for item in line_lst]
            line = ' '.join(list(map(str, line_lst)))
            print(line)

if __name__ == '__main__':
    ss = StateValueRVSimulate()
    ss.init_policy()
    ss.eval_policy()