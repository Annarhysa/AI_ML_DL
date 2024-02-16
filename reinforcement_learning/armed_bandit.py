import numpy as np

class Bandit:
    def __init__(self, num_arms):
        self.num_arms = num_arms
        self.q_true = np.random.normal(0, 1, num_arms)
        self.q_estimates = np.zeros(num_arms)
        self.arm_counts = np.zeros(num_arms)

    def choose_action(self, epsilon):
        if np.random.random() < epsilon:
            return np.random.randint(self.num_arms)  # Explore
        else:
            return np.argmax(self.q_estimates)  # Exploit

    def update(self, action, reward):
        self.arm_counts[action] += 1
        self.q_estimates[action] += (reward - self.q_estimates[action]) / self.arm_counts[action]

def bandit_simulation(num_arms, num_steps, epsilon):
    bandit = Bandit(num_arms)
    rewards = []

    for _ in range(num_steps):
        action = bandit.choose_action(epsilon)
        reward = np.random.normal(bandit.q_true[action], 1)
        bandit.update(action, reward)
        rewards.append(reward)

    return rewards

if __name__ == "__main__":
    num_arms = 1
    num_steps = 1000
    epsilon = 0.1

    rewards = bandit_simulation(num_arms, num_steps, epsilon)
    print("Average reward:", np.mean(rewards))