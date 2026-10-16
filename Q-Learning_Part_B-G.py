import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np

alpha = 0.35 # Learning rate
gamma = 0.90 # Discount factor for future rewards
epsilon = 0.25 # Exploration rate
episodes = 100 # Number of training episodes
total_steps = 200 # Total training steps executed

def experiment(x):
    match x:
        case 1:
            baseline()
        case 2:
            learning_rate()
        case 3:
            discount_factor()
        case 4:
            exploration_rate()
        case 5:
            best_agent()
        case _:
            print("Unknown experiment number. Please enter a number between 1 and 5.")

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


def train(env, Q, episodes, alpha, gamma, epsilon):
    episode_rewards = []
    for episode in range(episodes):
        current_state, _ = env.reset()
        total_reward = 0
        done = False

        while not done:
            current_action = action(Q, current_state, epsilon)
            # Take current action and observe the next state and reward
            next_state, reward, terminated, truncated, _ = env.step(current_action)
            # Update the Q-table using the looks at the reward and next state
            update(Q, current_state, current_action, reward, next_state, terminated, alpha, gamma)
            current_state = next_state
            total_reward += reward

            if terminated or truncated:
                done = True

        episode_rewards.append(total_reward)

    return episode_rewards

def run_greedy(env, Q):
    current_state, _ = env.reset()
    total_reward = 0
    steps = 0
    done = False

    while not done:
         #Select the best action based on the current Q-table
        current_action = action(Q, current_state, 0.0)
        # Take the selected action and looks at the next state and reward
        current_state, reward, terminated, truncated, _ = env.step(current_action)
        total_reward += reward
        steps += 1
        done = terminated or truncated

    # Check for the goal state
    reached_goal = terminated and current_state == 47
    return total_reward, steps, reached_goal

def baseline():
    rewards = train(env, Q_table, episodes, alpha, gamma, epsilon)
    plt.plot(rewards, label="Episode reward")
    plt.axhline(y=-13, color="black", linestyle="--", label="Optimal reward (-13)")
    plt.xlabel("Episode")
    plt.ylabel("Total reward")
    plt.title("Q-learning on CliffWalking (baseline)")
    plt.legend()
    plt.savefig("results/baseline1.png")
    plt.show()

def learning_rate():
    for a in [0.1, 0.5, 0.9]:
        alpha = a
        np.random.seed(0)
        Q = np.zeros((n_states, n_actions))  
        rewards = train(env, Q, episodes, alpha, gamma, epsilon)

        smooth = np.convolve(rewards, np.ones(10) / 10, mode="valid")
        plt.plot(smooth, label=f"α = {a}")

    plt.axhline(y=-13, color="black", linestyle="--", label="Optimal reward (-13)")
    plt.xlabel("Episode")
    plt.ylabel("Total reward (10-episode average)")
    plt.title("Effect of learning rate α")
    plt.legend()
    plt.savefig("results/learning_rate.png")
    plt.show()

def discount_factor():
    for g in [0.1, 0.5, 0.9]:
        gamma = g
        np.random.seed(0)
        Q = np.zeros((n_states, n_actions))  
        rewards = train(env, Q, episodes, alpha, gamma, epsilon)

        smooth = np.convolve(rewards, np.ones(10) / 10, mode="valid")
        plt.plot(smooth, label=f"γ = {g}")

    plt.axhline(y=-13, color="black", linestyle="--", label="Optimal reward (-13)")
    plt.xlabel("Episode")
    plt.ylabel("Total reward (10-episode average)")
    plt.title("Effect of discount factor γ")
    plt.legend()
    plt.savefig("results/discount_factor.png")
    plt.show()

def exploration_rate():
    for e in [0.05, 0.25, 0.5]:
        np.random.seed(0)
        Q = np.zeros((n_states, n_actions))
        rewards = train(env, Q, episodes, alpha, gamma, e)

        smooth = np.convolve(rewards, np.ones(10) / 10, mode="valid")
        plt.plot(smooth, label=f"ε = {e}")

    plt.axhline(y=-13, color="black", linestyle="--", label="Optimal reward (-13)")
    plt.xlabel("Episode")
    plt.ylabel("Total reward (10-episode average)")
    plt.title("Effect of exploration rate ε")
    plt.legend()
    plt.savefig("results/exploration_rate.png")
    plt.show()

def best_agent():
    for a, g, e in [(0.35, 0.90, 0.25), (0.9, 0.90, 0.05)]:
        np.random.seed(0)
        Q = np.zeros((n_states, n_actions))
        rewards = train(env, Q, episodes, a, g, e)

        smooth = np.convolve(rewards, np.ones(10) / 10, mode="valid")
        plt.plot(smooth, label=f"α = {a}, γ = {g}, ε = {e}")

        eval_rewards = []
        eval_steps = []
        eval_successes = []
        for _ in range(30):
            r, s, success = run_greedy(env, Q)
            eval_rewards.append(r)
            eval_steps.append(s)
            eval_successes.append(success)

        print(f"α = {a}, γ = {g}, ε = {e}")
        print("Mean reward:", np.mean(eval_rewards))
        print("Success rate:", np.mean(eval_successes) * 100)
        print("Average steps:", np.mean(eval_steps))

    plt.axhline(y=-13, color="black", linestyle="--", label="Optimal reward (-13)")
    plt.xlabel("Episode")
    plt.ylabel("Total reward (10-episode average)")
    plt.title("Best configuration vs. baseline")
    plt.legend()
    plt.savefig("results/best_agent.png")
    plt.show()


print("Choose a experiment:")
print("  • 1: Baseline")
print("  • 2: Learning Rate")
print("  • 3: Discount Factor")
print("  • 4: Exploration Rate")
print("  • 5: Best Agent")
experiment_number = int(input("Enter experiment number (1-5): "))
experiment(experiment_number)