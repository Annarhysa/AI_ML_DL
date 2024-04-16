import numpy as np
ROWS = 3
COLS = 4

# Define possible actions
ACTIONS = ['LEFT', 'RIGHT', 'UP', 'DOWN']

# Define the transition probabilities for each action
P = {
    'LEFT': np.array([[0, 0, 0, 1],
                      [0, 0.1, 0.9, 0],
                      [0, 0.1, 0.1, 0.8]]),
    'RIGHT': np.array([[1, 0, 0, 0],
                       [0.8, 0.1, 0.1, 0],
                       [0, 0.9, 0.1, 0]]),
    'UP': np.array([[0, 1, 0, 0],
                    [0, 0.1, 0.9, 0],
                    [0.8, 0.1, 0, 0.1]]),
    'DOWN': np.array([[0.1, 0.9, 0, 0],
                      [0.1, 0.1, 0, 0.8],
                      [0, 0, 1, 0]])
}

# Define rewards
REWARDS = np.array([[0, 0, 0, 1],
                    [0, 0, 0, -1],
                    [0, 0, 0, 0]])

# Initialize value function
V = np.zeros((ROWS, COLS))

# Policy initialization (uniform random policy)
policy = np.ones((ROWS, COLS, len(ACTIONS))) / len(ACTIONS)

# Policy Improvement
def policy_improvement(V, policy):
    policy_stable = True
    for i in range(ROWS):
        for j in range(COLS):
            old_action = np.argmax(policy[i, j])
            action_values = []
            for a, action in enumerate(ACTIONS):
                action_value = np.sum(P[action][i, j] * (REWARDS + V))
                action_values.append(action_value)
            best_action = np.argmax(action_values)
            if old_action != best_action:
                policy_stable = False
            policy[i, j] = np.eye(len(ACTIONS))[best_action]
    return policy, policy_stable

# Value Iteration
def value_iteration():
    theta = 0.0001
    while True:
        delta = 0
        for i in range(ROWS):
            for j in range(COLS):
                v = V[i, j]
                action_values = []
                for a, action in enumerate(ACTIONS):
                    action_value = np.sum(P[action][i, j] * (REWARDS + V))
                    action_values.append(action_value)
                V[i, j] = max(action_values)
                delta = max(delta, abs(v - V[i, j]))
        if delta < theta:
            break

# Run Value Iteration
value_iteration()

# Print optimal value function
print("Optimal Value Function:")
print(V)

# Policy Improvement
policy_stable = False
iteration = 0
while not policy_stable:
    policy, policy_stable = policy_improvement(V, policy)
    iteration += 1

# Print optimal policy
print("\nOptimal Policy after", iteration, "iterations:")
for i in range(ROWS):
    for j in range(COLS):
        print(ACTIONS[np.argmax(policy[i, j])], end=' ')
    print()
