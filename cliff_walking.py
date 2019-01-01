import numpy as np
import random
import matplotlib.pyplot as plt

alpha = 0.5
epsilon = 0.1
gamma = 1
x_length = 12
y_length = 4

actionRewards = np.zeros((x_length, y_length, 4))
actionRewards[:, :, :] = -5.0
actionRewards[1:11, 1, 2] = -100.0

actionRewards[5:7, 1, 0] = -1.0
actionRewards[4:6, 2, 1] = -1.0
actionRewards[5:7, 3, 2] = -1.0
actionRewards[6:8, 2, 3] = -1.0

actionRewards[0, 0, 1] = -100.0

#########################
#action and rewards
def observe(x, y, action):
    goal = 0
    if x == x_length - 1 and y == 0:
        goal = 1
    if action == 0:
        y = y+1
    if action == 1:
        x = x+1
    if action == 2:
        y = y-1
    if action == 3:
        x = x-1

    x = max(0, x)
    x = min(x_length - 1, x)
    y = max(0, y)
    y = min(y_length - 1, y)

    if goal == 1:
        return x, y, -1
    if x > 0 and x < x_length - 1 and y == 0:
        return 0, 0, -100
    if x >= 5 and x <= 7 and y == 1 and action == 0:
        return x, y, -1
    if x >= 5 and x <= 7 and y == 3 and action == 2:
        return x, y, -1
    if x >= 4 and x <= 6 and y == 2 and action == 1:
        return x, y, -1
    if x >= 6 and x <= 8 and y == 2 and action == 3:
        return x, y, -1
    return x, y, -5

#########################

#行动之后的目的地函数
actionDestination = []
for i in range(0, 12):
    actionDestination.append([])
    for j in range(0, 4):
        destination = dict()
        destination[0] = [i, min(j+1,3)]
        destination[1] = [min(i+1,11), j]
        if 0 < i < 11 and j == 1:
            destination[2] = [0,0]
        else:
            destination[2] = [i, max(j - 1, 0)]
        destination[3] = [max(i-1,0), j]
        actionDestination[-1].append(destination)
actionDestination[0][0][1] = [0,0]
#
# for i in range(12):
#     for j in range(4):
#         for k in range(4):
#             x_next, y_next, reward = observe(i,j,k)
#             if [x_next,y_next] != actionDestination[i][j][k]:
#                 print("next: ",i,j,k,actionDestination[i][j][k],x_next,y_next)
#             if reward != actionRewards[i][j][k]:
#                 print("reward: ",i,j,k,actionRewards[i][j][k],reward)

#epsilon-greedy algorithm
def epsilon_policy(x,y,q,eps):
    t = random.randint(0,3)
    if random.random() < eps:
        a = t
    else:
        q_max = q[x][y][0]
        a_max = 0
        for i in range(4):
            if q[x][y][i] >= q_max:
                q_max = q[x][y][i]
                a_max = i
        a = a_max
    return a

def max_q(x,y,q):
    q_max = q[x][y][0]
    a_max = 0
    for i in range(4):
        if q[x][y][i] >= q_max:
            q_max = q[x][y][i]
            a_max = i
    a = a_max
    return a

def q_learning(q):
    runs = 20
    rewards = np.zeros([500])
    for j in range(runs):
        for i in range(500):
            reward_sum = 0
            x = 0
            y = 0
            while True:
                a = epsilon_policy(x,y,q,epsilon)
                x_next, y_next, reward = observe(x,y,a)
                a_next = max_q(x_next,y_next,q)
                reward_sum += reward
                q[x][y][a] += alpha*(reward + gamma*q[x_next][y_next][a_next]-q[x][y][a])
                if x == x_length - 1 and y==0:
                    break
                x = x_next
                y = y_next
            rewards[i] = rewards[i] + reward_sum
    rewards = rewards/runs
    plt.plot(range(len(rewards)), rewards, label="q learning")
    avg_rewards = []
    for i in range(9):
        avg_rewards.append(np.mean(rewards[:i+1]))
    for i in range(10,len(rewards)+1):
        avg_rewards.append(np.mean(rewards[i-10:i]))
    return avg_rewards

qq = np.zeros([12,4,4])
q_learning_rewards = q_learning(qq)



def OptimalPolicy(q):
    for j in range(y_length-1,-1,-1):
        for i in range(x_length):
            a = max_q(i,j,q)
            if a == 0:
                print("↑ ", end="")
            if a == 1:
                print("→ ", end="")
            if a == 2:
                print("↓ ", end="")
            if a == 3:
                print("← ", end="")
        print("")
    print("")
print("Q-learning Optimal Policy")
OptimalPolicy(qq)

def OptimalPath(q):
    x = 0
    y = 0
    path = np.zeros([x_length,y_length]) - 1
    end = 0
    exist = np.zeros([x_length,y_length])
    best_path_rewards = 0
    while (x != x_length-1 or y != 0) and end == 0:
        a = max_q(x,y,q)
        path[x][y] = a
        if exist[x][y] == 1:
            end = 1
        exist[x][y] = 1
        x,y,r = observe(x,y,a)
        best_path_rewards = best_path_rewards + r
    for j in range(y_length-1,-1,-1):
        for i in range(x_length):
            if i == x_length-1 and j == 0:
                print(" G ", end="")
                continue
            a = path[i,j]
            if a == -1:
                print(" 0 ", end="")
            elif a == 0:
                print(" ↑ ", end="")
            elif a == 1:
                print(" → ", end="")
            elif a == 2:
                print(" ↓ ", end="")
            elif a == 3:
                print(" ← ", end="")
        print("")

    print('\nThe best path return value is %d' % best_path_rewards)

OptimalPath(qq)

# plt.plot(range(len(sarsa_rewards)),sarsa_rewards,label="sarsa")
plt.plot(range(len(q_learning_rewards)),q_learning_rewards, label="q learning average")
plt.ylim(-200,0)
plt.legend(loc="lower right")
plt.show()