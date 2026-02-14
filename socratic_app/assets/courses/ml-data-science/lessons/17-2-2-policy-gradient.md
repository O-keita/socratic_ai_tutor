# Policy Gradient Methods

## Introduction

Policy gradient methods directly learn a parameterized policy without needing a value function. They optimize the policy parameters to maximize expected return.

## Why Policy Gradients?

```python
import numpy as np
import pandas as pd

print("=== VALUE-BASED VS POLICY-BASED ===")
print("""
VALUE-BASED (Q-Learning, DQN):
  - Learn Q(s, a)
  - Derive policy: π(s) = argmax Q(s, a)
  - Requires max over actions
  
POLICY-BASED:
  - Learn π(a|s; θ) directly
  - Parameterized policy (neural network)
  - Output: action probabilities

ADVANTAGES of Policy Gradients:
✓ Handles continuous actions naturally
✓ Can learn stochastic policies
✓ Better convergence properties
✓ Works when Q-function complex

DISADVANTAGES:
✗ High variance
✗ Often sample inefficient
✗ Can converge to local optima
""")
```

## Policy Parameterization

```python
print("\n=== PARAMETERIZED POLICIES ===")
print("""
For DISCRETE actions (softmax policy):
  π(a|s; θ) = exp(θ^T φ(s,a)) / Σ exp(θ^T φ(s,a'))
  
  Or with neural network:
  features = NN(s)
  π(a|s) = softmax(features)

For CONTINUOUS actions (Gaussian policy):
  μ(s), σ(s) = NN(s)  
  π(a|s) = N(a; μ(s), σ(s))
  
  Sample: a = μ(s) + σ(s) × ε, where ε ~ N(0, I)
""")

def softmax_policy(state_features, theta):
    """Discrete action policy using softmax"""
    logits = np.dot(theta, state_features)
    exp_logits = np.exp(logits - np.max(logits))  # Stability
    probs = exp_logits / np.sum(exp_logits)
    return probs

def sample_action(probs):
    """Sample action from probability distribution"""
    return np.random.choice(len(probs), p=probs)

# Example
state_features = np.array([0.5, 0.3, 0.2])
theta = np.random.randn(4, 3)  # 4 actions, 3 features

probs = softmax_policy(state_features, theta)
print("Action probabilities:", probs.round(3))
print("Sampled action:", sample_action(probs))
```

## The Policy Gradient Theorem

```python
print("\n=== POLICY GRADIENT THEOREM ===")
print("""
OBJECTIVE: Maximize expected return
  J(θ) = E_π[Σ γ^t r_t] = E_π[G]

GRADIENT:
  ∇J(θ) = E_π[∇log π(a|s;θ) × Q^π(s,a)]
  
Intuition:
  - ∇log π(a|s): Direction to make action more likely
  - Q^π(s,a): How good was that action?
  - High Q → increase probability of action
  - Low Q → decrease probability

SCORE FUNCTION TRICK:
  ∇π / π = ∇log π
  
  This allows using log probabilities for gradient computation.
""")

def compute_log_prob_gradient(probs, action, state_features):
    """Gradient of log probability for softmax policy"""
    # For softmax: ∇log π(a|s) = φ(s,a) - Σ π(a'|s)φ(s,a')
    # Simplified for demonstration
    grad = np.zeros_like(probs)
    grad[action] = 1 - probs[action]
    for a in range(len(probs)):
        if a != action:
            grad[a] = -probs[a]
    return grad

print("Score function ∇log π tells us how to adjust action probabilities")
```

## REINFORCE Algorithm

```python
print("\n=== REINFORCE ===")
print("""
Monte Carlo Policy Gradient

Algorithm:
1. Initialize policy parameters θ
2. For each episode:
   a. Generate trajectory: s_0, a_0, r_0, s_1, a_1, r_1, ...
   b. For each step t:
      G_t = Σ_{k=t} γ^{k-t} r_k  (return from step t)
      θ ← θ + α × ∇log π(a_t|s_t;θ) × G_t
   
Uses full returns (Monte Carlo)
Update at end of episode
""")

def reinforce_episode(env, policy_network, optimizer, gamma=0.99):
    """One episode of REINFORCE"""
    states, actions, rewards = [], [], []
    
    state = env.reset()
    done = False
    
    # Collect trajectory
    while not done:
        probs = policy_network(state)
        action = sample_action(probs)
        next_state, reward, done = env.step(action)
        
        states.append(state)
        actions.append(action)
        rewards.append(reward)
        
        state = next_state
    
    # Compute returns
    returns = []
    G = 0
    for r in reversed(rewards):
        G = r + gamma * G
        returns.insert(0, G)
    
    # Update policy
    for s, a, G in zip(states, actions, returns):
        log_prob = np.log(policy_network(s)[a])
        loss = -log_prob * G  # Negative for gradient ascent
        # optimizer.step(loss)
    
    return sum(rewards)

print("REINFORCE: Simple but high variance")
```

## Variance Reduction: Baseline

```python
print("\n=== BASELINE FOR VARIANCE REDUCTION ===")
print("""
PROBLEM: REINFORCE has high variance
  - Returns can vary wildly
  - Slow, unstable learning

SOLUTION: Subtract a baseline b(s)

∇J(θ) = E[∇log π(a|s;θ) × (Q(s,a) - b(s))]

Common baseline: Value function V(s)
  - Q(s,a) - V(s) = Advantage A(s,a)
  - "How much better is action a than average?"

The baseline doesn't change the expected gradient!
  E[∇log π × b(s)] = 0 (provable)
  
But it reduces variance significantly.
""")

def reinforce_with_baseline(states, actions, returns, policy_net, value_net, optimizer):
    """REINFORCE with baseline"""
    for s, a, G in zip(states, actions, returns):
        # Advantage = return - baseline
        baseline = value_net(s)
        advantage = G - baseline
        
        # Policy update
        log_prob = np.log(policy_net(s)[a])
        policy_loss = -log_prob * advantage
        
        # Value update (to improve baseline)
        value_loss = (G - baseline) ** 2
        
        # optimizer.step(policy_loss + value_loss)

print("Advantage function centers the gradient around zero")
print("This dramatically reduces variance")
```

## Actor-Critic

```python
print("\n=== ACTOR-CRITIC ===")
print("""
Combine policy gradient with value function learning

ACTOR: Policy network π(a|s;θ)
  - Decides what action to take
  - Updated using policy gradient

CRITIC: Value network V(s;w) or Q(s,a;w)
  - Evaluates how good states/actions are
  - Provides baseline/advantage estimates

Advantage Actor-Critic (A2C):
  A(s,a) = r + γV(s') - V(s)  (TD estimate)
  
  Actor update: θ ← θ + α × ∇log π(a|s;θ) × A(s,a)
  Critic update: w ← w + β × ∇(r + γV(s';w) - V(s;w))²

Benefits:
  - Lower variance than REINFORCE
  - Can update every step (not episodic)
  - More sample efficient
""")

class ActorCritic:
    def __init__(self, state_dim, action_dim, hidden_dim=64):
        # Shared layers or separate networks
        self.actor = self._build_policy(state_dim, action_dim, hidden_dim)
        self.critic = self._build_value(state_dim, hidden_dim)
    
    def _build_policy(self, s_dim, a_dim, h_dim):
        # Would be neural network in practice
        return {'weights': np.random.randn(h_dim, a_dim)}
    
    def _build_value(self, s_dim, h_dim):
        return {'weights': np.random.randn(h_dim, 1)}
    
    def get_action(self, state):
        # Policy forward pass
        probs = self._policy_forward(state)
        return np.random.choice(len(probs), p=probs)
    
    def update(self, state, action, reward, next_state, done, gamma=0.99):
        # TD estimate of advantage
        V = self._value_forward(state)
        V_next = 0 if done else self._value_forward(next_state)
        advantage = reward + gamma * V_next - V
        
        # Actor update
        # policy_loss = -log_prob * advantage
        
        # Critic update
        # value_loss = advantage ** 2
        
        return advantage

print("Actor-Critic is the foundation of modern policy gradient methods")
```

## PPO (Proximal Policy Optimization)

```python
print("\n=== PPO ===")
print("""
State-of-the-art policy gradient algorithm

KEY IDEA: Limit how much policy can change per update
  - Prevents destructively large updates
  - More stable training

Clipped objective:
  r(θ) = π(a|s;θ) / π(a|s;θ_old)  (probability ratio)
  
  L^CLIP = min(r(θ) × A, clip(r(θ), 1-ε, 1+ε) × A)
  
  ε typically 0.1 to 0.3

If advantage > 0 (good action):
  - Increase probability, but clip at 1+ε
  
If advantage < 0 (bad action):
  - Decrease probability, but clip at 1-ε

PPO is simpler than TRPO but often works as well or better.
""")

def ppo_objective(old_probs, new_probs, advantages, clip_epsilon=0.2):
    """Compute PPO clipped objective"""
    # Probability ratio
    ratio = new_probs / (old_probs + 1e-8)
    
    # Clipped ratio
    clipped_ratio = np.clip(ratio, 1 - clip_epsilon, 1 + clip_epsilon)
    
    # Take minimum (pessimistic bound)
    objective = np.minimum(ratio * advantages, clipped_ratio * advantages)
    
    return np.mean(objective)

# Example
old_probs = np.array([0.3, 0.4, 0.35])
new_probs = np.array([0.5, 0.3, 0.4])  # Policy has changed
advantages = np.array([1.0, -0.5, 0.2])

obj = ppo_objective(old_probs, new_probs, advantages)
print(f"PPO objective: {obj:.4f}")
```

## Key Points

- **Policy gradient**: Directly optimize parameterized policy
- **REINFORCE**: Monte Carlo policy gradient, high variance
- **Baseline**: Subtract V(s) to reduce variance (advantage)
- **Actor-Critic**: Combine policy and value learning
- **PPO**: Clipped objective for stable updates
- **Continuous actions**: Natural fit for policy gradients

## Reflection Questions

1. Why do policy gradients have high variance compared to value-based methods?
2. How does the baseline reduce variance without changing the expected gradient?
3. What makes PPO more stable than vanilla policy gradients?
