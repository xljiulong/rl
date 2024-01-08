import numpy as np
import pandas as pd
from roomba_env import GridWordEnv
from s4_2_state_value_rv_opt import StateValueRVSimulate

class QValueRVSimulate(StateValueRVSimulate):
    def __init__(self, n_width: int = 5, 
                 n_height: int = 5, 
                 u_size=40, 
                 default_reward: float = 0, 
                 default_type=0,
                 action_len = 4) -> None:
        
       super().__init__(n_width, n_height, u_size, 
                         default_reward, default_type)            

    def sumQ_nxt_state(self, s: int,  Q:pd.DataFrame):
        sum = 0
        for a in range(0, self.action_len):
            sum += self.pi[s][a] * Q.loc[s][str(a)]

        return sum

    def eval_policy(self, theta = 0.001, eval_round=100):
        Q = pd.DataFrame(
            np.zeros((self.n_height * self.n_width, self.action_len)),
            columns=list(map(str, range(0, self.action_len)))
        )

        gamma = 0.8
            
        for iter in range(0, eval_round):
            k = -1
            delta = 0
            for state in range(self.n_width * self.n_height):
                # v = V[state]
                q = 0
                if self.grids.get_type(state) == 1:
                    print(f'state {state} will not get evaled')
                    continue

                if self.grids.get_reward(state) > 0:
                    continue

                # 迭代
                for action in range(0, self.action_len):
                    nxt_sate = self.step_from_state(state, action, skip_type=True)
                    # if nxt_sate == state:
                    #     continue

                    reward = self.get_reward(nxt_sate)

                    if self.grids.get_type(nxt_sate) == 1:
                        q = reward + gamma * (self.sumQ_nxt_state(state, Q)) 
                    else:
                        q = reward + gamma * (self.sumQ_nxt_state(nxt_sate, Q))
                    
                    delta = max(delta, np.abs(q - Q.loc[state][action]))
                    Q.loc[state][action] = q
                
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
    ss = QValueRVSimulate()
    ss.init_policy()
    ss.eval_policy()