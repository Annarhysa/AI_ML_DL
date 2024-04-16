import numpy as np

class MDP:
    def __init__(self, num_states, num_actions, transition_probs, rewards, gamma):
        self.num_states = num_states
        self.num_actions = num_actions
        self.transition_probs = transition_probs  # Array of shape (num_states, num_actions, num_states)
        self.rewards = rewards  # Array of shape (num_states, num_actions)
        self.gamma = gamma  # Discount factor

    def value_iteration(self, tol=1e-6):
        V = np.zeros(self.num_states)  # Initialize value function
        while True:
            V_new = np.zeros(self.num_states)
            for s in range(self.num_states):
                max_q_value = float('-inf')
                for a in range(self.num_actions):
                    q_value = np.sum(self.transition_probs[s, a] * (self.rewards[s, a] + self.gamma * V))
                    if q_value > max_q_value:
                        max_q_value = q_value
                V_new[s] = max_q_value
            if np.max(np.abs(V - V_new)) < tol:
                break
            V = V_new
        return V

if __name__ == "__main__":
    # Example MDP
    num_states = 3
    num_actions = 2
    transition_probs = np.array([[[0.7, 0.3, 0.0], [0.1, 0.8, 0.1]], [[0.0, 0.2, 0.8], [0.4, 0.4, 0.2]], [[1.0, 0.0, 0.0], [0.0, 0.0, 1.0]]])
    rewards = np.array([[1, 2], [3, 4], [0, 0]])
    gamma = 0.9

    mdp = MDP(num_states, num_actions, transition_probs, rewards, gamma)
    optimal_values = mdp.value_iteration()
    print("Optimal value function:", optimal_values)