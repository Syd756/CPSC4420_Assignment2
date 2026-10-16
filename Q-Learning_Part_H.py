import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np


alpha = 0.35 # Learning rate
gamma = 0.90 # Discount factor for future rewards
epsilon = 0.25 # Exploration rate
episodes = 100 # Number of training episodes
total_steps = 200 # Total training steps executed

print("Choose a mode:")
print("  • 1: No rendering")
print("  • 2: Rendering")
mode = input("Enter mode (1 or 2): ")
if mode == "1":
    env = gym.make("CliffWalking-v1", max_episode_steps=total_steps)
elif mode == "2":
    env = gym.make("CliffWalking-v1", render_mode="human", max_episode_steps=total_steps)
    env.unwrapped.metadata["render_fps"] = 60
else:
    print("Invalid mode. Please enter 1 or 2.")
    exit()

n_states = env.observation_space.n
n_actions = env.action_space.n

# Create a Q-table initialized to zeros 48x4
Q_table = np.zeros((n_states, n_actions))

def action(Q, current_state, epsilon):
    if np.random.rand() < epsilon:
        return np.random.randint(n_actions)
    else:
        # Picks the action with the highest Q-value for the current state
        # If two or more actions have the same maximum Q-value, randomly select one of them
        q = Q[current_state]
        best_actions = np.flatnonzero(q == q.max())
        return np.random.choice(best_actions)

def noisy_action(chosen_action, p):
    if np.random.rand() < p:
        # Randomly select an action that is not the chosen action
        other_actions = [a for a in range(n_actions) if a != chosen_action]
        return np.random.choice(other_actions)
    return chosen_action

def update(Q, current_state, action, reward, next_state, terminated, alpha, gamma):
    if terminated:
        # Goal has been reached
        td_target = reward
    else:
        # Calculate the TD target using the Bellman equation
        best_next_value = np.max(Q[next_state])
        td_target = reward + gamma * best_next_value
    # Update the Q-value for the current state-action pair using the TD target and learning rate
    Q[current_state][action] += alpha * (td_target - Q[current_state][action])

def train(env, Q, episodes, alpha, gamma, epsilon, p=0):
    episode_rewards = []
    for episode in range(episodes):
        current_state, _ = env.reset()
        total_reward = 0
        done = False

        while not done:
            # Select the next action based on the current Q-table and exploration rate
            current_action = action(Q, current_state, epsilon)
            # Take the selected action including noise and looks at the next state and reward
            taken_action = noisy_action(current_action, p)
            next_state, reward, terminated, truncated, _ = env.step(taken_action)
            # Update the Q-table using the observed reward and next state
            update(Q, current_state, taken_action, reward, next_state, terminated, alpha, gamma)
            current_state = next_state
            total_reward += reward

            if terminated or truncated:
                done = True

        episode_rewards.append(total_reward)

    return episode_rewards

def run_greedy(env, Q, p=0):
    current_state, _ = env.reset()
    total_reward = 0
    steps = 0
    done = False

    while not done:
        # Select the best action based on the current Q-table
        current_action = action(Q, current_state, 0.0)
        # Take the selected action including noise and looks at the next state and reward
        taken_action = noisy_action(current_action, p)
        current_state, reward, terminated, truncated, _ = env.step(taken_action)
        total_reward += reward
        steps += 1
        done = terminated or truncated

    reached_goal = terminated and current_state == 47
    return total_reward, steps, reached_goal


def evaluate(env, Q, p, n_episodes=30):
    eval_rewards = []
    eval_steps = []
    eval_successes = []
    for _ in range(n_episodes):
        r, s, success = run_greedy(env, Q, p)
        eval_rewards.append(r)
        eval_steps.append(s)
        eval_successes.append(success)

    print("  Mean reward:", np.mean(eval_rewards), "+/-", np.std(eval_rewards))
    print("  Success rate:", np.mean(eval_successes) * 100)
    print("  Average steps:", np.mean(eval_steps))

def noise_experiment():
    a, g, e = 0.9, 0.90, 0.05
    trained = {}

    for p in [0, 0.10, 0.25]:
        np.random.seed(0)
        Q = np.zeros((n_states, n_actions))
        rewards = train(env, Q, episodes, a, g, e, p)
        trained[p] = Q

        smooth = np.convolve(rewards, np.ones(10) / 10, mode="valid")
        plt.plot(smooth, label=f"p = {p}")

        print(f"Trained and evaluated with p = {p}:")
        evaluate(env, Q, p)

    plt.axhline(y=-13, color="black", linestyle="--", label="Optimal reward (-13)")
    plt.xlabel("Episode")
    plt.ylabel("Total reward (10-episode average)")
    plt.title("Effect of action noise p")
    plt.legend()
    plt.savefig("results/noise_experiment.png")
    plt.show()

    for p in [0.10, 0.25]:
        print(f"Trained with p = 0, evaluated with p = {p}:")
        evaluate(env, trained[0], p)

noise_experiment()