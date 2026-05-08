"""
drl_agent.py — Fast NumPy version (no sklearn MLP)
===================================================
Deep Q-Network using pure NumPy neural network.
No tensorflow, no pytorch, no sklearn MLP.
Runs in under 60 seconds on any CPU.
"""

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import joblib, os, json
from collections import deque

OUTPUT_DIR = "artifacts"
os.makedirs(OUTPUT_DIR, exist_ok=True)

ACTION_APPROVE  = 0
ACTION_REJECT   = 1
ACTION_ESCALATE = 2
ACTION_NAMES    = {0: "APPROVE", 1: "REJECT", 2: "ESCALATE"}


# ── Pure NumPy Neural Network ─────────────────────────────────────────────────
class NumpyMLP:
    """
    Lightweight 2-layer MLP using pure NumPy.
    Forward pass + backprop — no external ML libraries needed.
    ~100x faster than sklearn MLPRegressor for this use case.
    """
    def __init__(self, input_dim, hidden=32, output_dim=3, lr=0.005):
        # Xavier initialization
        self.W1 = np.random.randn(input_dim, hidden) * np.sqrt(2.0 / input_dim)
        self.b1 = np.zeros(hidden)
        self.W2 = np.random.randn(hidden, output_dim) * np.sqrt(2.0 / hidden)
        self.b2 = np.zeros(output_dim)
        self.lr = lr

    def relu(self, x):
        return np.maximum(0, x)

    def relu_grad(self, x):
        return (x > 0).astype(float)

    def forward(self, X):
        self.z1 = X @ self.W1 + self.b1
        self.a1 = self.relu(self.z1)
        self.z2 = self.a1 @ self.W2 + self.b2
        return self.z2   # Q-values (no activation on output)

    def predict(self, X):
        return self.forward(X)

    def train_step(self, X, targets):
        # Forward
        out = self.forward(X)
        # MSE loss gradient
        delta2 = 2 * (out - targets) / len(X)
        # Backprop layer 2
        dW2 = self.a1.T @ delta2
        db2 = delta2.sum(axis=0)
        # Backprop layer 1
        delta1 = (delta2 @ self.W2.T) * self.relu_grad(self.z1)
        dW1    = X.T @ delta1
        db1    = delta1.sum(axis=0)
        # Gradient descent
        self.W2 -= self.lr * dW2
        self.b2 -= self.lr * db2
        self.W1 -= self.lr * dW1
        self.b1 -= self.lr * db1
        return float(np.mean((out - targets) ** 2))


# ── Reward function ───────────────────────────────────────────────────────────
def compute_reward(action, true_label, fraud_prob):
    if action == ACTION_APPROVE:
        return +1.0 if true_label == 0 else -5.0
    elif action == ACTION_REJECT:
        return +10.0 if true_label == 1 else -2.0
    elif action == ACTION_ESCALATE:
        uncertainty = 1 - abs(fraud_prob - 0.5) * 2
        return (+4.0 if true_label == 1 else +1.0) * uncertainty
    return 0.0


# ── Environment ───────────────────────────────────────────────────────────────
class FraudEnvironment:
    def __init__(self, X, y, fraud_probs):
        self.X = X; self.y = y; self.fraud_probs = fraud_probs
        self.n = len(X); self.idx = 0

    def reset(self):
        perm = np.random.permutation(self.n)
        self.X = self.X[perm]; self.y = self.y[perm]
        self.fraud_probs = self.fraud_probs[perm]
        self.idx = 0
        return self._state()

    def _state(self):
        x     = self.X[self.idx]
        prob  = self.fraud_probs[self.idx]
        uncert= 1 - abs(prob - 0.5) * 2
        return np.append(x, [prob, uncert])

    def step(self, action):
        true_label = int(self.y[self.idx])
        fraud_prob = float(self.fraud_probs[self.idx])
        reward     = compute_reward(action, true_label, fraud_prob)
        self.idx  += 1
        done       = self.idx >= self.n
        next_state = self._state() if not done else np.zeros(self.X.shape[1] + 2)
        return next_state, reward, done, {"true_label": true_label, "action": action}


# ── DQN Agent ─────────────────────────────────────────────────────────────────
class DQNAgent:
    def __init__(self, state_dim, n_actions=3):
        self.n_actions    = n_actions
        self.epsilon      = 1.0
        self.epsilon_min  = 0.05
        self.epsilon_decay= 0.99
        self.gamma        = 0.95
        self.batch_size   = 32
        self.memory       = deque(maxlen=3000)
        self.q_network    = NumpyMLP(state_dim, hidden=32, output_dim=n_actions, lr=0.005)

    def remember(self, s, a, r, ns, done):
        self.memory.append((s, a, r, ns, done))

    def act(self, state):
        if np.random.rand() < self.epsilon:
            return np.random.randint(self.n_actions)
        q = self.q_network.predict(state.reshape(1, -1))[0]
        return int(np.argmax(q))

    def replay(self):
        if len(self.memory) < self.batch_size:
            return
        idx     = np.random.choice(len(self.memory), self.batch_size, replace=False)
        samples = [self.memory[i] for i in idx]

        states      = np.array([s[0] for s in samples])
        actions     = np.array([s[1] for s in samples])
        rewards     = np.array([s[2] for s in samples])
        next_states = np.array([s[3] for s in samples])
        dones       = np.array([s[4] for s in samples])

        current_q = self.q_network.predict(states)
        next_q    = self.q_network.predict(next_states)

        targets = current_q.copy()
        for i in range(self.batch_size):
            targets[i, actions[i]] = (rewards[i] if dones[i]
                                      else rewards[i] + self.gamma * np.max(next_q[i]))

        self.q_network.train_step(states, targets)

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def save(self, path):
        joblib.dump(self, path)


# ── Train ─────────────────────────────────────────────────────────────────────
def train_drl_agent(n_episodes=30):
    from data_preprocessing import load_data, preprocess

    print("[INFO] Loading data for DRL training...")
    df = load_data()
    X_train, X_test, y_train, y_test, scaler = preprocess(df)
    X_train = np.array(X_train); X_test  = np.array(X_test)
    y_train = np.array(y_train); y_test  = np.array(y_test)

    best_model  = joblib.load(os.path.join(OUTPUT_DIR, "best_model.pkl"))
    train_probs = best_model.predict_proba(X_train)[:, 1]
    test_probs  = best_model.predict_proba(X_test)[:,  1]

    # Small sample for speed
    MAX_TRAIN  = 1500
    fraud_idx  = np.where(y_train == 1)[0]
    legit_idx  = np.where(y_train == 0)[0]
    legit_samp = np.random.choice(legit_idx,
                                   size=min(MAX_TRAIN, len(legit_idx)),
                                   replace=False)
    sel        = np.concatenate([fraud_idx[:500], legit_samp])
    np.random.shuffle(sel)
    X_s = X_train[sel]; y_s = y_train[sel]; p_s = train_probs[sel]

    state_dim = X_s.shape[1] + 2
    agent     = DQNAgent(state_dim=state_dim)
    env       = FraudEnvironment(X_s, y_s, p_s)

    episode_rewards = []
    episode_actions = []

    print(f"[INFO] Training DRL Agent for {n_episodes} episodes...")
    for episode in range(1, n_episodes + 1):
        state = env.reset()
        total_r = 0; actions = {0:0,1:0,2:0}; done = False

        while not done:
            action = agent.act(state)
            next_s, r, done, info = env.step(action)
            agent.remember(state, action, r, next_s, done)
            agent.replay()
            state = next_s; total_r += r; actions[action] += 1

        episode_rewards.append(total_r)
        episode_actions.append(actions)

        if episode % 5 == 0:
            avg_r = np.mean(episode_rewards[-5:])
            print(f"  Episode {episode:3d}/{n_episodes} | "
                  f"Avg Reward: {avg_r:8.1f} | "
                  f"ε: {agent.epsilon:.3f} | "
                  f"Approve:{actions[0]} Reject:{actions[1]} Escalate:{actions[2]}")

    # Evaluate
    print("\n[INFO] Evaluating on test set...")
    TEST_SIZE  = 2000
    fraud_t    = np.where(y_test == 1)[0]
    legit_t    = np.random.choice(np.where(y_test == 0)[0],
                                   size=min(TEST_SIZE - len(fraud_t), len(np.where(y_test==0)[0])),
                                   replace=False)
    sel_t      = np.concatenate([fraud_t, legit_t])
    test_env   = FraudEnvironment(X_test[sel_t], y_test[sel_t], test_probs[sel_t])
    state      = test_env.reset()
    done       = False; total_reward = 0
    true_labels= []; all_actions = []
    agent.epsilon = 0  # No exploration during eval

    while not done:
        action = agent.act(state)
        next_s, r, done, info = test_env.step(action)
        state = next_s; total_reward += r
        true_labels.append(info["true_label"])
        all_actions.append(info["action"])

    all_actions   = np.array(all_actions)
    action_counts = {ACTION_NAMES[a]: int((all_actions == a).sum()) for a in range(3)}
    escalation_r  = action_counts["ESCALATE"] / len(all_actions) * 100

    print(f"\n{'='*50}")
    print("  DRL Agent Results")
    print(f"{'='*50}")
    print(f"  Total Test Reward : {total_reward:.1f}")
    print(f"  Actions           : {action_counts}")
    print(f"  Escalation Rate   : {escalation_r:.1f}%")

    # Plots
    _plot_rewards(episode_rewards)
    _plot_actions(episode_actions)
    _plot_pie(action_counts)

    metrics = {
        "total_test_reward": round(total_reward, 2),
        "action_counts":     action_counts,
        "escalation_rate":   round(escalation_r, 2),
        "n_episodes":        n_episodes,
    }
    with open(os.path.join(OUTPUT_DIR, "drl_metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    agent.save(os.path.join(OUTPUT_DIR, "drl_agent.pkl"))
    print("\n[DONE] DRL agent training complete.")
    return agent, metrics, episode_rewards


def _plot_rewards(rewards):
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(rewards, alpha=0.4, color="#00BCD4", linewidth=1, label="Episode reward")
    window = min(5, len(rewards))
    ma     = pd.Series(rewards).rolling(window).mean()
    ax.plot(ma, color="#FFD166", linewidth=2.5, label=f"{window}-ep moving avg")
    ax.set_xlabel("Episode"); ax.set_ylabel("Total Reward")
    ax.set_title("DRL Agent — Reward per Episode", fontsize=12, fontweight="bold")
    ax.legend(); ax.grid(alpha=0.2)
    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, "drl_training_rewards.png")
    plt.savefig(path, dpi=150); plt.close()
    print(f"[INFO] Saved → {path}")


def _plot_actions(episode_actions):
    approves  = [e[0] for e in episode_actions]
    rejects   = [e[1] for e in episode_actions]
    escalates = [e[2] for e in episode_actions]
    fig, ax   = plt.subplots(figsize=(10, 4))
    ep = range(1, len(episode_actions)+1)
    ax.stackplot(ep, approves, rejects, escalates,
                 labels=["APPROVE","REJECT","ESCALATE"],
                 colors=["#2ECC71","#E74C3C","#F39C12"], alpha=0.8)
    ax.set_xlabel("Episode"); ax.set_ylabel("Actions")
    ax.set_title("DRL Agent — Action Distribution", fontsize=12, fontweight="bold")
    ax.legend(loc="upper right"); ax.grid(alpha=0.2)
    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, "drl_action_distribution.png")
    plt.savefig(path, dpi=150); plt.close()
    print(f"[INFO] Saved → {path}")


def _plot_pie(action_counts):
    fig, ax = plt.subplots(figsize=(5, 5))
    labels  = list(action_counts.keys())
    sizes   = list(action_counts.values())
    colors  = ["#2ECC71","#E74C3C","#F39C12"]
    wedges, texts, autotexts = ax.pie(
        sizes, labels=labels, colors=colors,
        autopct="%1.1f%%", startangle=90,
        pctdistance=0.8,
        wedgeprops={"edgecolor":"white","linewidth":2}
    )
    for t in autotexts: t.set_fontweight("bold")
    ax.set_title("DRL Agent — Action Breakdown", fontsize=11, fontweight="bold")
    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, "drl_action_pie.png")
    plt.savefig(path, dpi=150); plt.close()
    print(f"[INFO] Saved → {path}")


if __name__ == "__main__":
    agent, metrics, rewards = train_drl_agent(n_episodes=30)
