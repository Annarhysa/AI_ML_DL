import numpy as np

class Environment:
    def __init__(self):
        self.states = ['S', 'A', 'B', 'C', 'G']  # States: S - Start, A, B, C, G - Goal
        self.num_states = len(self.states)
        self.actions = ['left', 'right']  # Actions: left, right
        self.num_actions = len(self.actions)
        self.transitions = {
            'S': {'left': 'S', 'right': 'A'},
            'A': {'left': 'S', 'right': 'B'},
            'B': {'left': 'A', 'right': 'C'},
            'C': {'left': 'B', 'right': 'G'},
            'G': {}
        }  # Transition function
        self.rewards = {
            'S': {'left': 0, 'right': 0},
            'A': {'left': 0, 'right': 0},
            'B': {'left': 0, 'right': 0},
            'C': {'left': 0, 'right': 0},
            'G': {}
        }  # Reward function

    def reset(self):
        return 'S'  # Reset to initial state 'S'

    def step(self, state, action):
        next_state = self.transitions[state][action]
        reward = self.rewards[state][action]
        done = (next_state == 'G')  # Terminates if reaching the goal state 'G'
        return next_state, reward, done







class MonteCarloOffPolicyControl:
    def __init__(self, env, num_episodes, gamma, behavior_policy, target_policy, epsilon):
        self.env = env
        self.num_episodes = num_episodes
        self.gamma = gamma
        self.behavior_policy = behavior_policy
        self.target_policy = target_policy
        self.epsilon = epsilon
        self.q_values = {(s, a): 0 for s in env.states for a in env.actions}
        self.c_values = {(s, a): 0 for s in env.states for a in env.actions}

    def generate_episode(self):
        episode = []
        state = self.env.reset()
        while True:
            action = np.random.choice(self.env.actions, p=self.behavior_policy[state])
            next_state, reward, done = self.env.step(state, action)
            episode.append((state, action, reward))
            state = next_state
            if done:
                break
        return episode

    def update_q_values(self, episode):
        G = 0
        weight = 1
        for t in reversed(range(len(episode))):
            state, action, reward = episode[t]
            G = self.gamma * G + reward
            self.c_values[(state, action)] += weight
            self.q_values[(state, action)] += (weight / self.c_values[(state, action)]) * (G - self.q_values[(state, action)])
            if action != self.target_policy[state]:
                break
            weight /= self.behavior_policy[state][action]

    def run(self):
        for _ in range(self.num_episodes):
            episode = self.generate_episode()
            self.update_q_values(episode)

# Example usage:
env = Environment()
# Define behavior policy (e.g., epsilon-greedy)
behavior_policy = {'S': [0.5, 0.5], 'A': [0.5, 0.5], 'B': [0.5, 0.5], 'C': [0.5, 0.5], 'G': []}
# Define target policy (e.g., greedy)
target_policy = {'S': 1, 'A': 1, 'B': 1, 'C': 1, 'G': []}
mc_off_policy_control = MonteCarloOffPolicyControl(env, num_episodes=1000, gamma=0.9, behavior_policy=behavior_policy, target_policy=target_policy, epsilon=0.1)
mc_off_policy_control.run()
print("Q Values:", mc_off_policy_control.q_values)
