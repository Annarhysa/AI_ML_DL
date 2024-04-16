import numpy as np

# Define the grid game environment
# 'S': Start, 'G': Goal, 'X': Obstacle
# The agent can move up, down, left, or right
grid = [
    ['S', ' ', ' ', ' '],
    [' ', 'X', ' ', 'X'],
    [' ', ' ', ' ', ' '],
    ['X', ' ', 'X', 'G']
]
num_rows = len(grid)
num_cols = len(grid[0])

# Define actions: up, down, left, right
actions = ['up', 'down', 'left', 'right']
num_actions = len(actions)

# Define rewards
rewards = {
    'S': 0,  # Start
    ' ': 0,  # Empty cell
    'X': -1, # Obstacle
    'G': 1   # Goal
}

# Define parameters
learning_rate = 0.1
discount_factor = 0.9
num_episodes = 1000
epsilon = 0.1

# Initialize Q-table
Q = np.zeros((num_rows, num_cols, num_actions))

# Q-learning algorithm
for episode in range(num_episodes):
    # Start state
    state = (0, 0)

    # Iterate until reaching the goal state
    while grid[state[0]][state[1]] != 'G':
        # Choose action using epsilon-greedy policy
        if np.random.rand() < epsilon:
            action = np.random.choice(actions)
        else:
            action = actions[np.argmax(Q[state[0], state[1]])]

        # Move to next state based on action
        next_state = state
        if action == 'up' and state[0] > 0:
            next_state = (state[0] - 1, state[1])
        elif action == 'down' and state[0] < num_rows - 1:
            next_state = (state[0] + 1, state[1])
        elif action == 'left' and state[1] > 0:
            next_state = (state[0], state[1] - 1)
        elif action == 'right' and state[1] < num_cols - 1:
            next_state = (state[0], state[1] + 1)

        # Update Q-value using Bellman equation
        reward = rewards[grid[next_state[0]][next_state[1]]]
        Q[state[0], state[1], actions.index(action)] += learning_rate * (reward +
            discount_factor * np.max(Q[next_state[0], next_state[1]]) - Q[state[0], state[1], actions.index(action)])

        # Move to next state
        state = next_state

# Define a function to run the trained agent in the environment and collect performance metrics
def evaluate_agent(Q, grid, num_episodes=100):
    total_rewards = []
    total_steps = []

    for _ in range(num_episodes):
        state = (0, 0)
        episode_reward = 0
        num_steps = 0

        while grid[state[0]][state[1]] != 'G':
            action = actions[np.argmax(Q[state[0], state[1]])]
            next_state = state

            # Move to next state based on action
            if action == 'up' and state[0] > 0:
                next_state = (state[0] - 1, state[1])
            elif action == 'down' and state[0] < num_rows - 1:
                next_state = (state[0] + 1, state[1])
            elif action == 'left' and state[1] > 0:
                next_state = (state[0], state[1] - 1)
            elif action == 'right' and state[1] < num_cols - 1:
                next_state = (state[0], state[1] + 1)

            # Update episode reward and number of steps
            episode_reward += rewards[grid[next_state[0]][next_state[1]]]
            num_steps += 1

            # Move to next state
            state = next_state

        total_rewards.append(episode_reward)
        total_steps.append(num_steps)

    avg_reward = np.mean(total_rewards)
    avg_steps = np.mean(total_steps)
    return avg_reward, avg_steps


# Analyze the results (e.g., Q-values)
print("Final Q-values:")
print(Q)

# Evaluate the performance of the trained agent
avg_reward, avg_steps = evaluate_agent(Q, grid)
print("Average reward:", avg_reward)
print("Average number of steps:", avg_steps)