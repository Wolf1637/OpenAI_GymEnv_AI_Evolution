import numpy as np
import gym

steps = 500
copies = 50
input1 = 8
output1 = 4
action = np.zeros(4, int)

w1 = np.random.rand(copies, input1, output1)
b1 = np.random.rand(copies, output1)
w1Sorted = w1
b1Sorted = b1
rewArr = np.zeros(copies)
rewArr[:] = -10000
generation = 0
maxAvgRew = -10000

env = gym.make('LunarLander-v2')
# print(env.action_space)
# print(env.observation_space)
while True:
    rewAvg = 0
    w1[int(copies / 2):copies - 1, :, :] = np.add(w1[0:int(copies / 2 - 1), :, :], np.random.normal(0.0, 0.1))
    b1[int(copies / 2):copies - 1,:] = np.add(b1[0:int(copies / 2 - 1),:], np.random.normal(0.0, 0.01))

    for co in range(copies):
        rewSum = 0
        render = False
        observation = env.reset()

        if generation % 30 == 0:
            if co % 40 == 0:
                render = True

        for st in range(steps):
            if render:
                env.render()
            decision = np.tanh(np.add(np.matmul(observation, w1[co, :, :]), b1[co, :]))
            if decision[0] > decision[1] and decision[0] > decision[2] and decision[0] > decision[3]:
                action = 0
            elif decision[1] > decision[2] and decision[1] > decision[3]:
                action = 1
            elif decision[2] > decision[3]:
                action = 2
            else:
                action = 3
            observation, reward, done, info = env.step(action)
            rewSum += reward
            if done:
                # print("Copy {0} finished after {1} timesteps".format(co + 1, st + 1))
                rewArr[co] = rewSum
                break
    rewAvg = np.average(rewArr)
    if rewAvg > maxAvgRew:
        maxAvgRew = rewAvg

    for i in range(copies):
        w1Sorted[i] = w1[rewArr.argmax()]
        b1Sorted[i] = b1[rewArr.argmax()]
        rewArr[rewArr.argmax()] = -10000
    w1 = w1Sorted
    b1 = b1Sorted
    print("Generation {0} finished with {1}/{2} average reward.".format(generation + 1, rewAvg, maxAvgRew))
    generation += 1
