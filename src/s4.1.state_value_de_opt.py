import numpy as np

from roomba_env import GridWordEnv

env = GridWordEnv()

word_h = 5
word_w = 5
length = word_h * word_w
gamma = 0.8
state = [i for i in range(length)]

action = ['n', 's', 'w', 'e']
ds_action = {'n': -word_w, 'e': 1, 's': word_w, 'w': -1}
policy = np.zeros([length, len(action)])

t_sque = [[n * word_w + e for e in range(0, word_w)] for n in range(word_h - 1, -1, -1)]
sque = []
for lt in t_sque:
    sque.extend(lt)  # [20, 21, 22, 23, 24, 15, 16, 17, 18, 19, 10, 11, 12, 13, 14, 5, 6, 7, 8, 9, 0, 1, 2, 3, 4]

# define reward
def reward(s):
    if s == 20:
        return 1
    if s == 12:
        return -10
    if s == 9:
        return 3
    else:
        return 0
    

def get_action(a):
    if a == 'n':
        return 0
    elif a == 'e':
        return 3
    elif a == 's':
        return 1
    elif a == 'w':
        return 2
    
def next_states(s, a):
    # next_sta
    if (s < word_w and a == 'n') or \
        (s % word_w == 0 and a == 'w') or \
        (s > length - word_w - 1 and  a == 's') or \
        ((s + 1) % word_w == 0 and a == 'e'):
        next_state = s
    else:
        next_state = s + ds_action[a]
    
    return next_state


def getsuccessor(s):
    '''
    返回所有的下一个可能状态的编号
    '''
    successor = []
    for a in action:
        if s == next_states(s, a):
            continue
        else:
            next = next_states(s, a)
            successor.append(next)

    return successor

def init_policy():
    for s in range(length):
        for a in action:
            if next_states(s, a) == s:
                continue
            new_action = get_action(a)
            policy[s][new_action] = 1/len(getsuccessor(s))


def policy_eval(theta = 0.0001):
    V = np.zeros(length)

    iter = 0
    
    while True:
        k = -1
        delta = 0
        for s in sque:
            k += 1
            if s in [9, 20, 12]:
                continue

            v = 0
            # print(f'第{k}的状态')
            for a in action:
                new_action = get_action(a)
                next_state = next_states(s, a)

                rewards = reward(next_state)
                if next_state == 12:
                    v += policy[s][new_action] * (rewards + gamma * V[s])
                else:
                    v += policy[s][new_action] * (rewards + gamma * V[next_state])
                print(f'v={v}')

            delta = max(delta, np.abs(v - V[s]))
            V[s] = v
        value = np.array(V).reshape(word_h, word_w)
        iter += 1

        print(f'k={iter}')
        print(f'current V is:')
        print(np.round(value, decimals=3))
        if delta < theta:
            break

        return V
    init_policy()
    value = policy_eval()