# Introduction to Reinforcement Learning

## Introduction

Reinforcement Learning (RL) is a paradigm where agents learn to make decisions by interacting with an environment, receiving rewards or penalties for their actions.

## The RL Framework

```python
import numpy as np
import pandas as pd

print("=== REINFORCEMENT LEARNING FRAMEWORK ===")
print("""
Core components:

AGENT: The learner/decision-maker
ENVIRONMENT: The world the agent interacts with
STATE (s): Current situation
ACTION (a): What the agent does
REWARD (r): Feedback signal
POLICY (π): Strategy for choosing actions

The RL Loop:
┌─────────────────────────────────────┐
│                                     │
│    Agent                            │
│      │                              │
│      │ Action a_t                   │
│      ▼                              │
│  Environment                        │
│      │                              │
│      │ State s_{t+1}, Reward r_t    │
│      ▼                              │
│    Agent                            │
│      │                              │
│      │ Action a_{t+1}               │
│      ▼                              │
│     ...                             │
└─────────────────────────────────────┘

Goal: Maximize cumulative reward over time
""")
```

## Key Concepts

```python
print("\n=== KEY CONCEPTS ===")
print("""
1. POLICY (π):
   Mapping from states to actions
   π(a|s) = P(Action = a | State = s)
   
   - Deterministic: a = π(s)
   - Stochastic: π(a|s) = probability

2. REWARD:
   Immediate feedback signal
   R(s, a, s') = reward for transition
   
   Goal: Maximize CUMULATIVE reward
   G_t = r_t + γr_{t+1} + γ²r_{t+2} + ...

3. DISCOUNT FACTOR (γ):
   0 ≤ γ ≤ 1
   - γ = 0: Only immediate reward matters
   - γ = 1: Future rewards equally important
   - Typical: γ = 0.99

4. VALUE FUNCTION:
   Expected return from a state
   V^π(s) = E[G_t | S_t = s]
   
5. ACTION-VALUE (Q-function):
   Expected return taking action a from state s
   Q^π(s, a) = E[G_t | S_t = s, A_t = a]
""")

def calculate_return(rewards, gamma=0.99):
    """Calculate discounted return"""
    G = 0
    for r in reversed(rewards):
        G = r + gamma * G
    return G

# Example: rewards over 5 timesteps
rewards = [1, 0, 1, 1, 10]
for gamma in [0.0, 0.5, 0.99]:
    G = calculate_return(rewards, gamma)
    print(f"γ={gamma}: Return = {G:.2f}")
```

## Types of RL Problems

```python
print("\n=== RL PROBLEM TYPES ===")
print("""
1. EPISODIC vs CONTINUING:
   - Episodic: Clear start and end (games)
   - Continuing: Goes on forever (stock trading)

2. MODEL-BASED vs MODEL-FREE:
   - Model-based: Learn environment dynamics
   - Model-free: Learn policy/value directly

3. ON-POLICY vs OFF-POLICY:
   - On-policy: Learn from current policy
   - Off-policy: Learn from different policy

4. VALUE-BASED vs POLICY-BASED:
   - Value-based: Learn Q-function, derive policy
   - Policy-based: Learn policy directly
   - Actor-Critic: Learn both

Examples:
┌──────────────────┬─────────────────────────┐
│ Problem          │ Type                    │
├──────────────────┼─────────────────────────┤
│ Chess            │ Episodic, Discrete      │
│ Robot walking    │ Continuing, Continuous  │
│ Atari games      │ Episodic, Discrete      │
│ Stock trading    │ Continuing, Continuous  │
└──────────────────┴─────────────────────────┘
""")
```

## Markov Decision Process (MDP)

```python
print("\n=== MARKOV DECISION PROCESS ===")
print("""
Formal framework for RL:

MDP = (S, A, P, R, γ)

S: State space
A: Action space
P: Transition probability P(s'|s, a)
R: Reward function R(s, a, s')
γ: Discount factor

MARKOV PROPERTY:
  Future depends only on current state
  P(s_{t+1} | s_t, a_t, s_{t-1}, ...) = P(s_{t+1} | s_t, a_t)
  
  "The current state contains all relevant information"
""")

# Simple MDP example: Grid World
print("""
Grid World Example:

  ┌───┬───┬───┬───┐
  │ S │   │   │ G │
  ├───┼───┼───┼───┤
  │   │ X │   │   │
  ├───┼───┼───┼───┤
  │   │   │   │   │
  └───┴───┴───┴───┘

S = Start, G = Goal (+10), X = Obstacle (-1)
Actions: Up, Down, Left, Right
States: Grid positions
Reward: -0.1 per step, +10 at goal, -1 hitting obstacle
""")
```

## Bellman Equations

```python
print("\n=== BELLMAN EQUATIONS ===")
print("""
RECURSIVE definitions of value:

STATE VALUE:
V^π(s) = Σ_a π(a|s) × Σ_{s'} P(s'|s,a) × [R(s,a,s') + γV^π(s')]

ACTION VALUE:
Q^π(s,a) = Σ_{s'} P(s'|s,a) × [R(s,a,s') + γ Σ_{a'} π(a'|s') × Q^π(s',a')]

OPTIMAL VALUE (Bellman Optimality):
V*(s) = max_a Σ_{s'} P(s'|s,a) × [R(s,a,s') + γV*(s')]
Q*(s,a) = Σ_{s'} P(s'|s,a) × [R(s,a,s') + γ max_{a'} Q*(s',a')]

Key insight:
  Value of a state = immediate reward + discounted future value
""")

def bellman_update(V, rewards, transitions, gamma=0.99):
    """One Bellman backup for value function"""
    n_states = len(V)
    V_new = np.zeros(n_states)
    
    for s in range(n_states):
        for a in range(len(transitions[s])):
            for s_next, prob in transitions[s][a]:
                V_new[s] = max(V_new[s], 
                               prob * (rewards[s][a] + gamma * V[s_next]))
    
    return V_new

print("Bellman equations enable iterative value computation.")
```

## Exploration vs Exploitation

```python
print("\n=== EXPLORATION VS EXPLOITATION ===")
print("""
THE DILEMMA:

EXPLOIT: Use known good actions
  → Get reliable rewards
  → Might miss better options

EXPLORE: Try new actions
  → Discover better strategies
  → Risk lower immediate reward

STRATEGIES:

1. ε-GREEDY:
   - With prob ε: random action (explore)
   - With prob 1-ε: best known action (exploit)
   - ε typically decays over time

2. SOFTMAX/BOLTZMANN:
   - P(a) ∝ exp(Q(s,a) / τ)
   - τ = temperature (high = more random)

3. UCB (Upper Confidence Bound):
   - Choose action with highest Q + exploration bonus
   - Bonus decreases with more samples
""")

def epsilon_greedy(Q_values, epsilon):
    """Select action using epsilon-greedy"""
    if np.random.random() < epsilon:
        return np.random.randint(len(Q_values))  # Explore
    return np.argmax(Q_values)  # Exploit

def softmax_action(Q_values, temperature=1.0):
    """Select action using softmax distribution"""
    exp_Q = np.exp(Q_values / temperature)
    probs = exp_Q / np.sum(exp_Q)
    return np.random.choice(len(Q_values), p=probs)

# Example
Q = np.array([1.0, 2.0, 1.5])  # Q-values for 3 actions
print("Q-values:", Q)
print(f"ε-greedy (ε=0.1): Action {epsilon_greedy(Q, 0.1)}")
print(f"Softmax (τ=0.5): Action {softmax_action(Q, 0.5)}")
```

## RL Applications

```python
print("\n=== RL APPLICATIONS ===")
print("""
GAMES:
  - AlphaGo/AlphaZero: Beat world champions
  - Atari: Deep Q-Network (DQN)
  - StarCraft: AlphaStar
  - Dota 2: OpenAI Five

ROBOTICS:
  - Robot manipulation
  - Autonomous navigation
  - Drone control
  - Locomotion (walking, running)

REAL-WORLD:
  - Recommendation systems
  - Ad placement
  - Traffic control
  - Energy optimization
  - Chip design (Google)
  - RLHF for LLMs (ChatGPT)

CHALLENGES:
  - Sample inefficiency
  - Sim-to-real transfer
  - Safety constraints
  - Reward engineering
""")
```

## Key Points

- **RL framework**: Agent, environment, state, action, reward
- **Goal**: Maximize cumulative discounted reward
- **MDP**: Formal model with Markov property
- **Bellman equations**: Recursive value definitions
- **Exploration-exploitation**: Balance trying new vs using known
- **Policy**: Mapping from states to actions

## Reflection Questions

1. Why is the discount factor important for infinite-horizon problems?
2. How does the Markov property simplify the learning problem?
3. When would you prefer exploration over exploitation?
