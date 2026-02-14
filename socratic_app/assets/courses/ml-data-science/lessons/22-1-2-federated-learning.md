# Federated Learning

## Introduction

Federated Learning enables training ML models on decentralized data without moving the data to a central server, preserving privacy while leveraging distributed datasets.

## Why Federated Learning?

```python
import numpy as np
import pandas as pd

print("=== THE FEDERATED LEARNING PROBLEM ===")
print("""
TRADITIONAL ML:
  Collect all data → Central server → Train model
  
PROBLEMS:
  1. PRIVACY: Can't/shouldn't share sensitive data
     - Medical records
     - Financial data
     - Personal messages
     
  2. REGULATIONS: Data must stay in jurisdiction
     - GDPR in EU
     - HIPAA in healthcare
     - Banking regulations
     
  3. SCALE: Data too large to move
     - Billions of devices
     - Terabytes per device
     
  4. OWNERSHIP: Organizations won't share
     - Competitive advantage
     - Liability concerns

FEDERATED LEARNING SOLUTION:
  Keep data where it is → Send model to data → Aggregate updates
""")
```

## Federated Averaging

```python
print("\n=== FEDAVG ALGORITHM ===")
print("""
Core algorithm for federated learning (McMahan et al., 2017)

ALGORITHM:
1. Server initializes global model w_0
2. For each round t = 1, 2, ...:
   a. Select subset of K clients
   b. Send global model w_t to selected clients
   c. Each client k:
      - Train on local data for E epochs
      - Get updated weights w_k
   d. Server aggregates:
      w_{t+1} = Σ (n_k/n) × w_k
      where n_k = samples at client k, n = total samples

KEY PARAMETERS:
  - K: Number of clients per round
  - E: Local epochs
  - B: Local batch size
  - η: Learning rate
""")

class FedAvgServer:
    """Federated Averaging server"""
    
    def __init__(self, initial_weights):
        self.global_weights = initial_weights.copy()
        self.round = 0
    
    def select_clients(self, clients, fraction=0.1):
        """Select subset of clients for this round"""
        k = max(1, int(len(clients) * fraction))
        return np.random.choice(clients, k, replace=False)
    
    def aggregate(self, client_updates):
        """Aggregate client updates weighted by data size"""
        total_samples = sum(n for _, n in client_updates)
        
        # Weighted average
        new_weights = np.zeros_like(self.global_weights)
        for weights, n_samples in client_updates:
            new_weights += (n_samples / total_samples) * weights
        
        self.global_weights = new_weights
        self.round += 1
        
        return self.global_weights

class FedAvgClient:
    """Federated learning client"""
    
    def __init__(self, client_id, local_data, local_labels):
        self.client_id = client_id
        self.data = local_data
        self.labels = local_labels
        self.n_samples = len(local_data)
    
    def local_train(self, global_weights, epochs=5, lr=0.01):
        """Train on local data"""
        weights = global_weights.copy()
        
        for _ in range(epochs):
            # Simple gradient descent (simplified)
            predictions = self.data @ weights
            errors = predictions - self.labels
            gradient = self.data.T @ errors / self.n_samples
            weights = weights - lr * gradient
        
        return weights, self.n_samples

# Demo
np.random.seed(42)
n_features = 5

# Create clients with different data distributions
clients = []
for i in range(10):
    # Each client has different amount of data
    n = np.random.randint(50, 200)
    X = np.random.randn(n, n_features)
    # Different clients may have different label distributions
    true_weights = np.random.randn(n_features)
    y = X @ true_weights + np.random.randn(n) * 0.1
    clients.append(FedAvgClient(i, X, y))

# Initialize server
server = FedAvgServer(np.zeros(n_features))

# Run federated learning
print("Federated Learning simulation:")
for round_num in range(5):
    # Select clients
    selected_ids = server.select_clients(list(range(10)), fraction=0.3)
    
    # Collect updates
    updates = []
    for cid in selected_ids:
        client = clients[cid]
        updated_weights, n_samples = client.local_train(
            server.global_weights, epochs=3
        )
        updates.append((updated_weights, n_samples))
    
    # Aggregate
    new_global = server.aggregate(updates)
    
    print(f"  Round {round_num + 1}: {len(selected_ids)} clients, "
          f"weight norm = {np.linalg.norm(new_global):.4f}")
```

## Privacy in Federated Learning

```python
print("\n=== PRIVACY CONSIDERATIONS ===")
print("""
FEDERATED ≠ PRIVATE by default

RISKS:
1. Model updates can leak information
   - Membership inference
   - Gradient inversion attacks
   
2. Memorization in model
   - Model might memorize training data
   
3. Inference attacks
   - Deduce properties of training data

ADDITIONAL PRIVACY TECHNIQUES:

1. DIFFERENTIAL PRIVACY:
   Add noise to gradients before sending
   
   noisy_gradient = gradient + Laplace(0, scale)
   
   - Provides mathematical privacy guarantees
   - Trade-off: noise reduces model quality

2. SECURE AGGREGATION:
   Cryptographic protocols so server only sees sum
   
   - Server can't see individual updates
   - Uses secret sharing, homomorphic encryption
   
3. TRUSTED EXECUTION:
   Run aggregation in secure enclave (SGX)
   
   - Hardware-based protection
   - Limited availability
""")

def add_differential_privacy(gradient, epsilon=1.0, sensitivity=1.0):
    """Add Laplacian noise for differential privacy"""
    scale = sensitivity / epsilon
    noise = np.random.laplace(0, scale, gradient.shape)
    return gradient + noise

# Demo
gradient = np.array([0.5, -0.3, 0.1, 0.8, -0.2])
print("Differential privacy on gradients:")
print(f"  Original gradient: {gradient}")

for eps in [0.1, 1.0, 10.0]:
    noisy = add_differential_privacy(gradient, epsilon=eps)
    print(f"  ε={eps}: {noisy.round(3)}")
```

## Non-IID Data Challenges

```python
print("\n=== NON-IID DATA ===")
print("""
CHALLENGE: Client data is not identically distributed

TYPES OF NON-IID:
1. Label distribution skew
   Client A: mostly cats
   Client B: mostly dogs
   
2. Feature distribution skew
   Client A: young users
   Client B: older users
   
3. Quantity skew
   Client A: 1000 samples
   Client B: 10 samples

PROBLEMS:
  - Slower convergence
  - Lower final accuracy
  - Client drift (local optima)
  - Some clients may hurt model

SOLUTIONS:

1. DATA SHARING:
   Share small public dataset
   
2. FEDPROX:
   Add regularization toward global model
   L_local + μ/2 ||w - w_global||²
   
3. SCAFFOLD:
   Use control variates to correct drift
   
4. PERSONALIZATION:
   Learn both global and local models
""")

class FedProxClient:
    """FedProx client with proximal term"""
    
    def __init__(self, client_id, local_data, local_labels, mu=0.01):
        self.client_id = client_id
        self.data = local_data
        self.labels = local_labels
        self.n_samples = len(local_data)
        self.mu = mu  # Proximal term weight
    
    def local_train(self, global_weights, epochs=5, lr=0.01):
        """Train with proximal regularization"""
        weights = global_weights.copy()
        
        for _ in range(epochs):
            # Standard gradient
            predictions = self.data @ weights
            errors = predictions - self.labels
            gradient = self.data.T @ errors / self.n_samples
            
            # Add proximal term: μ(w - w_global)
            proximal_term = self.mu * (weights - global_weights)
            
            weights = weights - lr * (gradient + proximal_term)
        
        return weights, self.n_samples

print("FedProx adds proximal regularization to prevent client drift")
```

## Federated Learning Systems

```python
print("\n=== FL SYSTEMS AND FRAMEWORKS ===")
print("""
1. TENSORFLOW FEDERATED (TFF):
   - Google's FL framework
   - Simulation and production
   - High-level API
   
   import tensorflow_federated as tff
   
   @tff.federated_computation
   def federated_train(model, data):
       return tff.federated_mean(
           tff.federated_map(local_train, [model, data])
       )

2. PYSYFT:
   - Privacy-preserving ML
   - Differential privacy, secure computation
   - PyTorch integration

3. FLOWER:
   - Framework-agnostic
   - Easy to use
   - Scalable

4. NVIDIA CLARA:
   - Healthcare focused
   - Medical imaging
   - Regulatory compliance

5. APPLE ON-DEVICE:
   - Keyboard predictions
   - Siri improvements
   - Billions of devices

DEPLOYMENT CONSIDERATIONS:
  - Device availability (mobile phones sleep)
  - Communication efficiency (compress updates)
  - Stragglers (timeouts)
  - Fault tolerance
""")
```

## Real-World Applications

```python
print("\n=== FL APPLICATIONS ===")
print("""
1. MOBILE KEYBOARD:
   Google Gboard: Next-word prediction
   - Billions of devices
   - Never see user typing
   
2. HEALTHCARE:
   Collaborative medical AI
   - Hospitals keep patient data
   - Share only model updates
   - FDA-compliant
   
3. FINANCIAL SERVICES:
   Fraud detection across banks
   - Banks can't share customer data
   - Improve models together
   
4. AUTONOMOUS VEHICLES:
   Learn from fleet
   - Each car has local data
   - Improve together

5. SMART HOME:
   Voice assistants
   - Personalization
   - Privacy-preserving
   
6. DRUG DISCOVERY:
   Pharmaceutical companies
   - Protect proprietary data
   - Collaborate on research
""")
```

## Key Points

- **Federated Learning**: Train on decentralized data
- **FedAvg**: Core algorithm with weighted averaging
- **Privacy**: FL alone isn't private; add DP, secure aggregation
- **Non-IID**: Real data is heterogeneous; needs special handling
- **Systems**: TFF, Flower, PySyft for implementation
- **Applications**: Mobile, healthcare, finance, autonomous vehicles

## Reflection Questions

1. When is federated learning better than centralized training?
2. How do you handle clients with very different data distributions?
3. What are the trade-offs between privacy and model quality in FL?
