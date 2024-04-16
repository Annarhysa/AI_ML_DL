import numpy as np

class Environment:
    def __init__(self):
        self.num_states = 3
        self.num_actions = 2
        self.transition_probs = np.array([[[0.5, 0.5, 0.0], [1.0, 0.0, 0.0]],
                                          [[0.0, 0.0, 1.0], [0.0, 0.0, 1.0]],
                                          [[0.0, 1.0, 0.0], [0.5, 0.5, 0.0]]])
        self.rewards = np.array([[1.0, 2.0], [0.0, 0.0], [5.0, -1.0]])

class QLearning:
    def __init__(self, env, alpha=0.1, epsilon=0.1, discount_factor=0.9):
        self.env = env
        self.Q = np.zeros((env.num_states, env.num_actions))
        self.alpha = alpha
        self.epsilon = epsilon
        self.discount_factor = discount_factor

    def choose_action(self, state):
        if np.random.rand() < self.epsilon:
            return np.random.choice(self.env.num_actions)
        else:
            return np.argmax(self.Q[state])

    def update(self, state, action, reward, next_state):
        next_action = np.argmax(self.Q[next_state])
        td_target = reward + self.discount_factor * self.Q[next_state, next_action]
        td_error = td_target - self.Q[state, action]
        self.Q[state, action] += self.alpha * td_error

# Example usage of Q-Learning
env = Environment()
q_learning = QLearning(env)
num_episodes = 1000
for _ in range(num_episodes):
    state = np.random.randint(env.num_states)
    while True:
        action = q_learning.choose_action(state)
        next_state = np.random.choice(env.num_states, p=env.transition_probs[state, action])
        reward = env.rewards[state, action]
        q_learning.update(state, action, reward, next_state)
        if next_state == 0:
            break
        state = next_state
print("Q-values after Q-Learning:")
print(q_learning.Q)
