import numpy as np

class Environment:
    def __init__(self):
        self.num_states = 3
        self.num_actions = 2
        self.transition_probs = np.array([[[0.5, 0.5, 0.0], [1.0, 0.0, 0.0]],
                                          [[0.0, 0.0, 1.0], [0.0, 0.0, 1.0]],
                                          [[0.0, 1.0, 0.0], [0.5, 0.5, 0.0]]])
        self.rewards = np.array([[1.0, 2.0], [0.0, 0.0], [5.0, -1.0]])

class QLearningWithLinearFunctionApproximation:
    def __init__(self, env, feature_dim, alpha=0.1, epsilon=0.1, discount_factor=0.9):
        self.env = env
        self.alpha = alpha
        self.epsilon = epsilon
        self.discount_factor = discount_factor
        self.weights = np.zeros((feature_dim, env.num_actions))

    def featurize_state(self, state):
        # For simplicity, we use a simple identity featurization here
        return np.eye(self.env.num_states)[state]

    def choose_action(self, state):
        if np.random.rand() < self.epsilon:
            return np.random.choice(self.env.num_actions)
        else:
            state_features = self.featurize_state(state)
            values = np.dot(state_features, self.weights)
            return np.argmax(values)

    def update(self, state, action, reward, next_state):
        state_features = self.featurize_state(state)
        next_state_features = self.featurize_state(next_state)
        q_values = np.dot(state_features, self.weights)
        next_q_values = np.dot(next_state_features, self.weights)
        td_target = reward + self.discount_factor * np.max(next_q_values)
        td_error = td_target - q_values[action]
        self.weights[:, action] += self.alpha * td_error * state_features

# Example usage of Q-Learning with Linear Function Approximation
env = Environment()
feature_dim = env.num_states  # Number of features is equal to the number of states in this example
q_learning = QLearningWithLinearFunctionApproximation(env, feature_dim)
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
print("Weights after Q-Learning with Linear Function Approximation:")
print(q_learning.weights)
