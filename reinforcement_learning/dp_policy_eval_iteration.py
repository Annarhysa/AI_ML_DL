import numpy as np

# Define the environment
num_states = 3
num_actions = 2
gamma = 0.9

# Define the transition probabilities
# P[state, action, next_state] = probability
P = np.array([
    [[0.7, 0.3, 0.0], [0.0, 1.0, 0.0]],
    [[0.8, 0.2, 0.0], [0.4, 0.4, 0.2]],
    [[1.0, 0.0, 0.0], [0.0, 0.0, 1.0]]
])

# Define the rewards
# R[state, action] = reward
R = np.array([
    [-1, 10],
    [-1, -1],
    [0, -1]
])

# Random policy
policy = np.array([[0.5, 0.5], [0.5, 0.5], [0.5, 0.5]])

def policy_evaluation(policy, P, R, gamma=0.9, theta=1e-6):
    V = np.zeros(num_states)
    while True:
        delta = 0
        for s in range(num_states):
            v = V[s]
            V[s] = sum(policy[s, a] * sum(P[s, a, s1] * (R[s, a] + gamma * V[s1]) for s1 in range(num_states)) for a in range(num_actions))
            delta = max(delta, abs(v - V[s]))
        if delta < theta:
            break
    return V

def policy_iteration(P, R, gamma=0.9, theta=1e-6):
    policy = np.ones((num_states, num_actions)) / num_actions
    while True:
        V = policy_evaluation(policy, P, R, gamma, theta)
        policy_stable = True
        for s in range(num_states):
            old_action = np.argmax(policy[s])
            new_action = np.argmax([sum(P[s, a, s1] * (R[s, a] + gamma * V[s1]) for s1 in range(num_states)) for a in range(num_actions)])
            if old_action != new_action:
                policy_stable = False
            policy[s] = np.eye(num_actions)[new_action]
        if policy_stable:
            break
    return policy, V

# Perform policy iteration
policy_optimal, V_optimal = policy_iteration(P, R, gamma)

print("Optimal Policy:")
print(policy_optimal)
print("\nOptimal Value Function:")
print(V_optimal)
