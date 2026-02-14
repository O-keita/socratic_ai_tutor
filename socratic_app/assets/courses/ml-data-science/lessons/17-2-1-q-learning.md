# Q-Learning

## Introduction

Q-Learning is a model-free reinforcement learning algorithm that learns the optimal action-value function (Q-function) through temporal difference learning.

## The Q-Function

```python
import numpy as np
import pandas as pd

print("=== THE Q-FUNCTION ===")
print("""
Q(s, a) = Expected return starting from state s, taking action a,
          then following the optimal policy

Q*(s, a) = max expected cumulative reward

Q-table: Store Q-values for all state-action pairs

         Action 0   Action 1   Action 2
State 0    2.5        1.2        0.8
State 1    0.5        3.1        1.5
State 2    1.0        0.9        2.8
...

Policy from Q:
  π*(s) = argmax_a Q*(s, a)
  "Choose action with highest Q-value"
""")
```

## Temporal Difference Learning

```python
print("\n=== TEMPORAL DIFFERENCE (TD) ===")
print("""
KEY IDEA: Learn from incomplete sequences

Monte Carlo: Wait for episode end
  V(s) ← V(s) + α(G_t - V(s))
  Problem: Must wait for episode to finish

TD Learning: Update after each step
  V(s) ← V(s) + α(r + γV(s') - V(s))
              └─────┬─────┘
              TD target

TD Error: δ = r + γV(s') - V(s)
  - Difference between estimate and target
  - Used to update value function

Benefits:
  - Learn online, step-by-step
  - Works for continuing tasks
  - Lower variance than Monte Carlo
""")

def td_update(V, s, r, s_next, alpha=0.1, gamma=0.99):
    """One TD(0) update"""
    td_target = r + gamma * V[s_next]
    td_error = td_target - V[s]
    V[s] = V[s] + alpha * td_error
    return V, td_error

# Example
V = np.array([0.0, 0.0, 1.0, 0.0])  # Initial values (state 2 is terminal)
s, r, s_next = 0, 0.5, 2

V, error = td_update(V, s, r, s_next)
print(f"TD update: V[0] = {V[0]:.3f}, TD error = {error:.3f}")
```

## Q-Learning Algorithm

```python
print("\n=== Q-LEARNING ALGORITHM ===")
print("""
Off-policy TD control

Update rule:
Q(s, a) ← Q(s, a) + α × [r + γ max_a' Q(s', a') - Q(s, a)]
                        └──────────────┬──────────────┘
                                  TD target

Algorithm:
1. Initialize Q(s, a) for all s, a
2. For each episode:
   a. Start in state s
   b. While not terminal:
      - Choose a using ε-greedy from Q
      - Take action a, observe r, s'
      - Update: Q(s,a) ← Q(s,a) + α[r + γ max Q(s',a') - Q(s,a)]
      - s ← s'
3. Repeat until convergence

KEY INSIGHT:
  - Learns optimal Q* regardless of policy used
  - "Off-policy": Can learn while following any policy
  - The max makes it greedy with respect to Q
""")

class QLearning:
    def __init__(self, n_states, n_actions, alpha=0.1, gamma=0.99, epsilon=0.1):
        self.Q = np.zeros((n_states, n_actions))
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
    
    def choose_action(self, state):
        """Epsilon-greedy action selection"""
        if np.random.random() < self.epsilon:
            return np.random.randint(self.Q.shape[1])
        return np.argmax(self.Q[state])
    
    def update(self, s, a, r, s_next, done):
        """Q-learning update"""
        if done:
            td_target = r
        else:
            td_target = r + self.gamma * np.max(self.Q[s_next])
        
        td_error = td_target - self.Q[s, a]
        self.Q[s, a] += self.alpha * td_error
        
        return td_error
    
    def get_policy(self):
        """Extract greedy policy from Q"""
        return np.argmax(self.Q, axis=1)

# Example usage
agent = QLearning(n_states=4, n_actions=2)
print("Initial Q-table:")
print(agent.Q)
```

## SARSA: On-Policy Alternative

```python
print("\n=== SARSA ===")
print("""
On-policy TD control

Update rule:
Q(s, a) ← Q(s, a) + α × [r + γ Q(s', a') - Q(s, a)]
                                   └─┬─┘
                          Next action a' from policy!

SARSA = State, Action, Reward, State', Action'

Difference from Q-Learning:
  Q-Learning: Uses max_a' Q(s', a') - optimal action
  SARSA: Uses Q(s', a') - actual next action from policy

SARSA is "safer":
  - Learns to avoid risky actions if exploration might hit them
  - Q-Learning assumes optimal play, might be riskier
""")

class SARSA:
    def __init__(self, n_states, n_actions, alpha=0.1, gamma=0.99, epsilon=0.1):
        self.Q = np.zeros((n_states, n_actions))
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
    
    def choose_action(self, state):
        if np.random.random() < self.epsilon:
            return np.random.randint(self.Q.shape[1])
        return np.argmax(self.Q[state])
    
    def update(self, s, a, r, s_next, a_next, done):
        """SARSA update - uses actual next action"""
        if done:
            td_target = r
        else:
            td_target = r + self.gamma * self.Q[s_next, a_next]
        
        td_error = td_target - self.Q[s, a]
        self.Q[s, a] += self.alpha * td_error
        
        return td_error

print("""
Comparison:

Q-Learning (off-policy):
  - Learns Q* regardless of behavior
  - Can be more sample efficient
  - Might overestimate in some cases

SARSA (on-policy):
  - Learns Q^π for current policy
  - Safer in cliff-walking scenarios
  - Policy must be exploratory
""")
```

## Grid World Example

```python
print("\n=== GRID WORLD EXAMPLE ===")
print("""
Simple 4x4 grid:

  ┌───┬───┬───┬───┐
  │ 0 │ 1 │ 2 │ 3 │
  ├───┼───┼───┼───┤
  │ 4 │ 5 │ 6 │ 7 │
  ├───┼───┼───┼───┤
  │ 8 │ 9 │10 │11 │
  ├───┼───┼───┼───┤
  │12 │13 │14 │15 │
  └───┴───┴───┴───┘

Start: 0, Goal: 15
Actions: 0=Up, 1=Right, 2=Down, 3=Left
Reward: -1 per step, +10 at goal
""")

def create_gridworld():
    """Create simple gridworld environment"""
    n_states = 16
    n_actions = 4  # Up, Right, Down, Left
    
    # Transition dynamics
    def get_next_state(s, a):
        row, col = s // 4, s % 4
        if a == 0: row = max(0, row - 1)      # Up
        elif a == 1: col = min(3, col + 1)    # Right
        elif a == 2: row = min(3, row + 1)    # Down
        elif a == 3: col = max(0, col - 1)    # Left
        return row * 4 + col
    
    def step(s, a):
        s_next = get_next_state(s, a)
        done = (s_next == 15)
        reward = 10 if done else -1
        return s_next, reward, done
    
    return step

# Train Q-learning agent
step_fn = create_gridworld()
agent = QLearning(n_states=16, n_actions=4, alpha=0.5, gamma=0.95, epsilon=0.3)

n_episodes = 500
for episode in range(n_episodes):
    s = 0  # Start state
    while True:
        a = agent.choose_action(s)
        s_next, r, done = step_fn(s, a)
        agent.update(s, a, r, s_next, done)
        s = s_next
        if done:
            break

print("Learned Q-values (reshaped as 4x4 for best action):")
print(np.max(agent.Q, axis=1).reshape(4, 4).round(1))

print("\nOptimal policy (0=Up, 1=Right, 2=Down, 3=Left):")
policy = agent.get_policy().reshape(4, 4)
arrows = {0: '↑', 1: '→', 2: '↓', 3: '←'}
for row in policy:
    print(' '.join(arrows[a] for a in row))
```

## Hyperparameters

```python
print("\n=== KEY HYPERPARAMETERS ===")
print("""
1. LEARNING RATE (α):
   - How much to update Q-values
   - Too high: Unstable, oscillates
   - Too low: Slow learning
   - Typical: 0.1 to 0.5
   - Can decay over time

2. DISCOUNT FACTOR (γ):
   - How much to value future rewards
   - γ = 0: Only immediate reward
   - γ = 0.99: Long-term planning
   - Higher γ = longer horizon

3. EXPLORATION (ε):
   - Balance exploration/exploitation
   - Start high (0.5-1.0)
   - Decay to low (0.01-0.1)
   - Annealing schedule important

4. TABLE INITIALIZATION:
   - Zeros: Neutral
   - Optimistic: Encourages exploration
   - Random: Can speed up early learning
""")

def epsilon_schedule(episode, max_episodes, start=1.0, end=0.01):
    """Linear decay of epsilon"""
    return max(end, start - (start - end) * episode / max_episodes)

print("Epsilon decay schedule:")
for ep in [0, 250, 500, 750, 1000]:
    eps = epsilon_schedule(ep, 1000)
    print(f"  Episode {ep}: ε = {eps:.2f}")
```

## Key Points

- **Q-function**: Expected return for state-action pairs
- **TD learning**: Update values using bootstrapped estimates
- **Q-Learning**: Off-policy, uses max for update
- **SARSA**: On-policy, uses actual next action
- **ε-greedy**: Balance exploration and exploitation
- **Convergence**: Guaranteed for tabular Q-learning (conditions apply)

## Reflection Questions

1. Why does Q-Learning use max while SARSA uses the actual next action?
2. How does the learning rate affect convergence?
3. When might SARSA outperform Q-Learning?
