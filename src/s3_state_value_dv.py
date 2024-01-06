import numpy as np

class sweeprobot():
    def __init__(self) -> None:
        self.S = [[[0, 0], [0, 2],[0, 3],[0, 4],[0, 5]],
                  [[1, 0], [1, 2],[1, 3],[1, 4],[1, 5]],
                  [[2, 0], [2, 2],[2, 3],[2, 4],[2, 5]],
                  [[3, 0], [3, 2],[3, 3],[3, 4],[3, 5]],
                  [[4, 0], [4, 2],[4, 3],[4, 4],[4, 5]],
                  [[5, 0], [5, 2],[5, 3],[5, 4],[5, 5]],
                ]
        
        self.A = [[None, None], [-1, 0], [1, 0], [0, -1], [0, 1]]

        self.V = [[None for i in range(6)] for j in range(6)]

        self.V[1][1] = 0
        self.V[5][4] = 0

        self.pi = None
        self.gamma = 0.8

    def reward(self, s, a):
        [truth1, truth2] = np.add(s, a) == [5, 4]
        [truth3, truth4] = np.add(s, a) == [1, 1]
        [truth5, truth6] = np.add(s, a) == [3, 3]

        # s 转移到5，4
        if s != [5, 4] and all([truth1, truth2]):
            return 3
        
        # s转移到充电
        if s != [1, 1] and all([truth3, truth4]):
            return 1
        
        # s 到障碍物
        if all([truth5, truth6]):
            return -10
        
        return 0
    
    def cal_coefficient(self):
        coef_matrix = [[0 for i in range(25)] for j in range(25)]
        b = [0 for i in range(25)]
        for i in range(1, 6):
            for j in range(1, 6):
                [truth1, truth2] = [i == 5, j == 4]
                [truth3, truth4] = [i == 1, j == 1]
                [truth5, truth6] = [i == 3, j == 3]


                if all([truth1, truth2]):
                    continue

                if all([truth3, truth4]):
                    continue

                if all([truth5, truth6]):
                    continue
                    
                
                count_action = 0
                if i - 1 >= 1: # 空间是1 - 5
                    count_action += 1
                
                if i + 1 <= 5:
                    count_action += 1

                if j - 1 >= 1:
                    count_action += 1
                
                if j + 5 <= 5:
                    count_action += 1
                
                self.pi = 1 / count_action # 计算策略

                b_value = 0
                coef_current_state = 0

                if i - 1 >= 1:
                    b_value = b_value + self.pi * self.reward(self.S[i][j], self.A[1])

