import numpy as np

from roomba_env import GridWordEnv

class StateValueSimulate(GridWordEnv):
    def __init__(self, n_width: int = 5, 
                 n_height: int = 5, 
                 u_size=40, 
                 default_reward: float = 0, 
                 default_type=0,
                 action_len = 4) -> None:
        
        super().__init__(n_width, n_height, u_size, 
                         default_reward, default_type)
        
        self.action_len = action_len
        self.pi = np.zeros([self.n_width * self.n_height, self.action_len])


    def get_reward(self, s_index: int):
        return self.grids.get_reward(s_index)
    
    def step(self, action: int):
        state, reward, done, info = super().step(action)
        return state


    def step_from_state(self, c_state:int, action: int):
        assert self.action_space.contains(action), f'{action} ({type(action)} invalid)'
        old_x, old_y = self._state_to_xy(c_state)
        new_x, new_y = old_x, old_y

        if action == 2:
            new_x -= 1
        elif action == 3:
            new_x += 1
        elif action == 0:
            new_y += 1
        elif action == 1:
            new_y -= 1

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

        state = self._xy_to_state((new_x, new_y))
       
        return state


    def get_successors(self, c_state: int):
        successors = []
        # TODO 验证获取全集 
        # self.action_space
        for a in range(0, self.action_len): # 可以验证下 action space 如何取全集
            nxt_state = self.step_from_state(c_state=c_state, action=a)
            if c_state == nxt_state:
                continue
            successors.append(nxt_state)

        return successors
            
        
    def init_policy(self):
        # pi = [[] for s in range(self.n_height * self.n_width)]

        for s in range(self.n_width * self.n_height):
            for a in range(self.action_len):
                if s == self.step_from_state(s, a):
                    continue
                action = a
                successor_num = len(self.get_successors(s))
                self.pi[s][a] = 1.0/successor_num

    def eval_policy(self, theta = 0.0001, eval_round=1000):
        V = np.zeros(self.n_width * self.n_height)
        gamma = 0.8

        for iter in range(0, eval_round):
            k = -1
            delta = 0
            for state in range(self.n_width * self.n_height):
                v = V[state]
                if self.grids.get_type(state) == 1:
                    print(f'state {state} will not get evaled')
                    continue

                # 迭代
                for action in range(0, self.action_len):
                    nxt_sate = self.step_from_state(state, action)
                    if nxt_sate == state:
                        continue

                    reward = self.get_reward(nxt_sate)

                    if self.grids.get_type(nxt_sate) == 1:
                        # V[state] += self.pi[state][action] * (reward + gamma * V[state])
                        pass
                    else:
                        V[state] += self.pi[state][action] * (reward + gamma * V[nxt_sate])
                    
            delta = max(delta, np.abs(v - V[state]))
            print(f'iter {iter} delta is {delta}')

        if delta <= theta:
            print(f'delta is good enough {delta}')
            exit(0)


if __name__ == '__main__':
    ss = StateValueSimulate()
    ss.init_policy()
    ss.eval_policy()