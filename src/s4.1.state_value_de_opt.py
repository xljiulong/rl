import numpy as np

from roomba_env import GridWordEnv

class StateValueSimulate(GridWordEnv):
    def __init__(self, n_width: int = 5, 
                 n_height: int = 5, 
                 u_size=40, 
                 default_reward: float = 0, 
                 default_type=0) -> None:
        
        super().__init__(n_width, n_height, u_size, 
                         default_reward, default_type)
        


