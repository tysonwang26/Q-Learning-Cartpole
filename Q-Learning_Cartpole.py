import gym
import numpy as np

env = gym.make('CartPole-v0')
env = env.unwrapped
N_DIVIDE = 4                      # state divide
N_STATE = N_DIVIDE ** 4           # q_table column size
N_ACTION = env.action_space.n     # cartpole has 2 action

EPSILON = 0.95
GAMMA = 0.9
ALPHA = 0.2                         # learn rate
TRAIN_TIME = 150                   # training times

q_table = np.random.uniform(low=-1, high=1, size=(N_STATE, N_ACTION))

def bins(clip_min, clip_max, num):
    return np.linspace(clip_min, clip_max, num + 1)[1:-1]

def digitize_state(observation):
    cart_pos, cart_v, pole_angle, pole_v = observation
    digitized = [np.digitize(cart_pos, bins(-2.4, 2.4, N_DIVIDE)),
                 np.digitize(cart_v, bins(-3.0, 3.0, N_DIVIDE)),
                 np.digitize(pole_angle, bins(-0.5, 0.5, N_DIVIDE)),
                 np.digitize(pole_v, bins(-2.0, 2.0, N_DIVIDE))]

    return sum([x * (N_DIVIDE ** i) for i, x in enumerate(digitized)])

def get_action(state, action, observation, reward):
    next_state = digitize_state(observation)
    if np.random.uniform() < EPSILON:
        next_action = np.argmax(q_table[next_state])
    else:
        next_action = np.random.choice([0, 1])

    q_table[state, action] = q_table[state, action] + ALPHA * (reward + GAMMA * q_table[next_state, next_action])
    return next_action, next_state

for episode in range(TRAIN_TIME):
    s = env.reset()
    state = digitize_state(s)
    a = np.argmax(q_table[state])

    while True:
        env.render()
        s, r, done, info = env.step(a)

        # ---------------------------------
        # x, x_dot, theta, theta_dot = s
        # r1 = (env.x_threshold - abs(x)) / env.x_threshold - 0.8
        # r2 = (env.theta_threshold_radians - abs(theta)) / env.theta_threshold_radians - 0.5
        # r = r1 + r2
        if done:
            r = -200
        # ---------------------------------

        a, state = get_action(state, a, s, r)
        if done:
            break
    print(episode)
env.close()