# Deep Reinforcement Learning

## Introduction

Deep RL combines deep neural networks with reinforcement learning, enabling agents to learn from high-dimensional inputs like images and handle complex environments.

## DQN: Deep Q-Network

```python
import numpy as np
import pandas as pd

print("=== DEEP Q-NETWORK (DQN) ===")
print("""
Combine Q-Learning with deep neural networks

PROBLEM: Q-table doesn't scale
  - Atari: 84×84×4 images = huge state space
  - Continuous states: Infinite
  - Need function approximation

SOLUTION: Approximate Q with neural network
  Q(s, a; θ) ≈ Q*(s, a)
  
  Input: State (e.g., image)
  Output: Q-values for all actions

Original DQN (2015):
  - Input: 84×84×4 stacked frames
  - Conv layers → FC layers → Q-values
  - Achieved human-level on Atari games
""")

print("""
DQN Architecture:

  Game Frame (84×84×4)
         ↓
  Conv2D(32, 8×8, stride 4)
         ↓
  Conv2D(64, 4×4, stride 2)
         ↓
  Conv2D(64, 3×3, stride 1)
         ↓
      Flatten
         ↓
    Dense(512)
         ↓
  Dense(n_actions)  ← Q-values
""")
```

## Key DQN Innovations

```python
print("\n=== DQN INNOVATIONS ===")
print("""
1. EXPERIENCE REPLAY:
   - Store transitions (s, a, r, s') in buffer
   - Sample random minibatches for training
   - Breaks correlation between consecutive samples
   - Reuse data multiple times
   
2. TARGET NETWORK:
   - Maintain two networks: Q and Q_target
   - Update Q_target periodically (every C steps)
   - Stabilizes training (target doesn't change constantly)
   
   TD target: y = r + γ max_a' Q_target(s', a')
   Loss: (Q(s, a) - y)²

Without these: Training is unstable and diverges!
""")

class ReplayBuffer:
    def __init__(self, capacity=100000):
        self.buffer = []
        self.capacity = capacity
        self.position = 0
    
    def push(self, state, action, reward, next_state, done):
        """Store transition"""
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        self.buffer[self.position] = (state, action, reward, next_state, done)
        self.position = (self.position + 1) % self.capacity
    
    def sample(self, batch_size):
        """Sample random batch"""
        indices = np.random.choice(len(self.buffer), batch_size, replace=False)
        batch = [self.buffer[i] for i in indices]
        return zip(*batch)
    
    def __len__(self):
        return len(self.buffer)

print("Experience replay buffer stores past experiences for reuse")
```

## DQN Training Loop

```python
print("\n=== DQN TRAINING ===")
print("""
def train_dqn(env, n_episodes):
    # Initialize networks
    Q = DQN(state_dim, action_dim)
    Q_target = copy(Q)
    
    buffer = ReplayBuffer()
    optimizer = Adam(Q.parameters(), lr=1e-4)
    
    for episode in range(n_episodes):
        state = env.reset()
        
        while not done:
            # Epsilon-greedy action
            if random() < epsilon:
                action = random_action()
            else:
                action = argmax(Q(state))
            
            # Execute and store
            next_state, reward, done = env.step(action)
            buffer.push(state, action, reward, next_state, done)
            
            # Train on batch
            if len(buffer) > batch_size:
                states, actions, rewards, next_states, dones = buffer.sample(batch_size)
                
                # Compute targets
                with no_grad():
                    targets = rewards + gamma * max(Q_target(next_states)) * (1 - dones)
                
                # Compute predictions
                predictions = Q(states)[actions]
                
                # Update
                loss = MSE(predictions, targets)
                optimizer.step(loss)
            
            # Update target network periodically
            if step % target_update_freq == 0:
                Q_target = copy(Q)
            
            state = next_state
""")
```

## DQN Improvements

```python
print("\n=== DQN IMPROVEMENTS ===")
print("""
1. DOUBLE DQN:
   Problem: Q-learning overestimates Q-values
   Solution: Use Q to select action, Q_target to evaluate
   
   y = r + γ Q_target(s', argmax_a Q(s', a))

2. DUELING DQN:
   Separate value and advantage streams:
   Q(s,a) = V(s) + A(s,a) - mean(A(s,·))
   
   Better generalization across actions

3. PRIORITIZED EXPERIENCE REPLAY:
   Sample important transitions more often
   Priority based on TD error
   
4. NOISY NETS:
   Parametric noise for exploration
   Replace ε-greedy
   
5. DISTRIBUTIONAL RL (C51):
   Learn distribution of returns, not just mean
   
6. RAINBOW:
   Combine all improvements above
   State-of-the-art DQN
""")

print("""
Performance comparison (Atari, median score):

DQN:        100% (baseline)
Double DQN: 111%
Dueling:    117%
PER:        128%
Rainbow:    230%

Rainbow combines: Double, Dueling, PER, n-step, 
                  distributional, noisy nets
""")
```

## Policy Gradient with Deep Networks

```python
print("\n=== DEEP POLICY GRADIENT ===")
print("""
PPO Architecture (Actor-Critic):

     State Input
          ↓
    Shared Layers (optional)
         / \\
        ↓   ↓
    Actor   Critic
      ↓       ↓
  π(a|s)   V(s)

Actor: Outputs action distribution
  - Discrete: Softmax over actions
  - Continuous: Mean and std of Gaussian

Critic: Outputs state value V(s)
  - Used for advantage estimation
  - Baseline to reduce variance
""")

print("""
PPO Training:

for iteration in range(n_iterations):
    # Collect trajectories with current policy
    trajectories = collect_rollouts(policy, env, n_steps)
    
    # Compute advantages (GAE or simple)
    advantages = compute_advantages(trajectories, value_net)
    
    # Multiple epochs on same data
    for epoch in range(n_epochs):
        for batch in trajectories.batches():
            # PPO clipped objective
            ratio = new_probs / old_probs
            clipped = clip(ratio, 1-ε, 1+ε)
            policy_loss = -min(ratio * adv, clipped * adv)
            
            # Value loss
            value_loss = (V(s) - returns)²
            
            # Entropy bonus (encourages exploration)
            entropy = -sum(p * log(p))
            
            total_loss = policy_loss + c1 * value_loss - c2 * entropy
            optimizer.step(total_loss)
""")
```

## Continuous Control

```python
print("\n=== CONTINUOUS ACTIONS ===")
print("""
For continuous action spaces (robotics, control):

1. Gaussian Policy:
   μ(s), σ(s) = Actor(s)
   a ~ N(μ(s), σ(s))
   
   log π(a|s) = -0.5 × ((a - μ)/σ)² - log(σ) - 0.5×log(2π)

2. SAC (Soft Actor-Critic):
   - Maximum entropy RL
   - Automatic temperature tuning
   - State-of-the-art for continuous control
   
   Objective: max E[Σ r + α × H(π)]
   where H(π) = entropy of policy

3. TD3 (Twin Delayed DDPG):
   - Two Q-networks (like Double DQN)
   - Delayed policy updates
   - Target policy smoothing
   - Good for robotics
""")

def gaussian_policy(state, actor_net):
    """Sample from Gaussian policy"""
    mean, log_std = actor_net(state)
    std = np.exp(log_std)
    
    # Reparameterization trick
    noise = np.random.randn(*mean.shape)
    action = mean + std * noise
    
    # Log probability
    log_prob = -0.5 * ((action - mean) / std)**2 - np.log(std) - 0.5 * np.log(2 * np.pi)
    log_prob = np.sum(log_prob, axis=-1)
    
    return action, log_prob

print("Continuous control is crucial for robotics applications")
```

## Sim-to-Real Transfer

```python
print("\n=== SIM-TO-REAL ===")
print("""
Train in simulation, deploy in real world

CHALLENGES:
  - Reality gap: Sim ≠ Real
  - Dynamics differ
  - Sensor noise
  - Unmodeled physics

SOLUTIONS:

1. DOMAIN RANDOMIZATION:
   - Randomize simulation parameters
   - Mass, friction, delays, visuals
   - Learn robust policies

2. DOMAIN ADAPTATION:
   - Learn to transform sim → real
   - Feature matching
   - Adversarial training

3. REAL-WORLD FINE-TUNING:
   - Pre-train in sim
   - Fine-tune with real data
   - Safe exploration important!

Examples:
  - OpenAI: Rubik's cube solving with robot hand
  - Google: Sim-to-real for legged robots
  - Boston Dynamics: Uses simulation extensively
""")
```

## Key Points

- **DQN**: Neural network approximates Q-function
- **Experience replay**: Store and reuse transitions
- **Target network**: Stabilize training with delayed updates
- **Double DQN**: Reduce overestimation
- **PPO/SAC**: State-of-the-art policy gradient methods
- **Continuous control**: Gaussian policies for robotics

## Reflection Questions

1. Why is experience replay important for stable DQN training?
2. How does the target network prevent instability?
3. What makes sim-to-real transfer challenging?
