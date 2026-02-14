# Introduction to Graph Neural Networks

## Introduction

Graph Neural Networks (GNNs) extend deep learning to graph-structured data, enabling learning on social networks, molecules, knowledge graphs, and more.

## Why Graphs?

```python
import numpy as np
import pandas as pd

print("=== GRAPH-STRUCTURED DATA ===")
print("""
Many real-world problems involve relationships:

SOCIAL NETWORKS:
  - Nodes: Users
  - Edges: Friendships/follows
  - Task: Friend recommendation, influence prediction

MOLECULES:
  - Nodes: Atoms
  - Edges: Chemical bonds
  - Task: Property prediction, drug discovery

KNOWLEDGE GRAPHS:
  - Nodes: Entities (people, places, things)
  - Edges: Relations (born_in, works_at)
  - Task: Link prediction, question answering

CITATION NETWORKS:
  - Nodes: Papers
  - Edges: Citations
  - Task: Topic classification

Standard neural networks can't handle:
  - Variable number of neighbors
  - No fixed ordering of nodes
  - Relational structure
""")
```

## Graph Representation

```python
print("\n=== GRAPH BASICS ===")
print("""
Graph G = (V, E, X)

V: Set of nodes (vertices)
E: Set of edges (connections)
X: Node features (optional)

Adjacency Matrix A:
  A[i,j] = 1 if edge between i and j
  A[i,j] = 0 otherwise
  
For undirected graphs: A = A^T (symmetric)

Example:
     1 --- 2
     |     |
     3 --- 4

A = [0 1 1 0]    Node features X:
    [1 0 0 1]    [x_1]
    [1 0 0 1]    [x_2]
    [0 1 1 0]    [x_3]
                 [x_4]
""")

# Create simple graph
def create_adjacency_matrix(edges, n_nodes):
    A = np.zeros((n_nodes, n_nodes))
    for i, j in edges:
        A[i, j] = 1
        A[j, i] = 1  # Undirected
    return A

edges = [(0, 1), (0, 2), (1, 3), (2, 3)]
A = create_adjacency_matrix(edges, 4)

print("Adjacency matrix:")
print(A.astype(int))

# Degree matrix
D = np.diag(np.sum(A, axis=1))
print("\nDegree matrix (diagonal):")
print(D.astype(int))
```

## Message Passing Framework

```python
print("\n=== MESSAGE PASSING ===")
print("""
Core idea: Nodes aggregate information from neighbors

GENERAL FRAMEWORK:
1. MESSAGE: Create messages from neighbors
   m_j→i = M(h_i, h_j, e_ij)
   
2. AGGREGATE: Combine all incoming messages
   m_i = AGG({m_j→i : j ∈ N(i)})
   
3. UPDATE: Update node representation
   h_i' = U(h_i, m_i)

After K layers: Node "sees" K-hop neighborhood

   Layer 0: Just the node itself
   Layer 1: Node + immediate neighbors
   Layer 2: Node + 2-hop neighborhood
   ...
""")

def simple_message_passing(A, X, W, num_layers=2):
    """Simple GNN with message passing"""
    H = X.copy()
    
    for layer in range(num_layers):
        # Aggregate neighbor features
        messages = A @ H  # Sum of neighbor features
        
        # Normalize by degree
        D_inv = np.diag(1 / (np.sum(A, axis=1) + 1e-8))
        normalized = D_inv @ messages
        
        # Update with transformation
        H = np.tanh(normalized @ W[layer])
        
        print(f"Layer {layer + 1} output shape: {H.shape}")
    
    return H

# Example
X = np.random.randn(4, 3)  # 4 nodes, 3 features
W = [np.random.randn(3, 3) for _ in range(2)]  # 2 layer weights

H = simple_message_passing(A, X, W)
print("Final node embeddings shape:", H.shape)
```

## Graph Convolutional Network (GCN)

```python
print("\n=== GCN ===")
print("""
Spectral approach with efficient approximation

GCN Layer:
  H' = σ(D̃^(-1/2) Ã D̃^(-1/2) H W)

Where:
  Ã = A + I (add self-loops)
  D̃ = degree matrix of Ã
  H = input features
  W = learnable weights
  σ = activation (ReLU)

Normalization D̃^(-1/2) Ã D̃^(-1/2):
  - Symmetric normalization
  - Prevents exploding/vanishing features
  - Equivalent to averaging neighbor features

Simplified:
  H' = σ(Â H W)
  
  where Â is normalized adjacency with self-loops
""")

def gcn_layer(A, H, W):
    """One GCN layer"""
    # Add self-loops
    A_hat = A + np.eye(A.shape[0])
    
    # Compute degree
    D_hat = np.diag(np.sum(A_hat, axis=1))
    
    # Symmetric normalization
    D_inv_sqrt = np.diag(1 / np.sqrt(np.diag(D_hat)))
    A_norm = D_inv_sqrt @ A_hat @ D_inv_sqrt
    
    # Message passing and transformation
    H_new = np.tanh(A_norm @ H @ W)
    
    return H_new

# Stack layers
H = X
for i, w in enumerate(W):
    H = gcn_layer(A, H, w)
    print(f"After GCN layer {i+1}: {H.shape}")
```

## GraphSAGE

```python
print("\n=== GraphSAGE ===")
print("""
Sample and Aggregate - scalable to large graphs

KEY IDEAS:
1. SAMPLING: Don't use all neighbors
   - Sample fixed number of neighbors
   - Enables mini-batch training
   
2. AGGREGATION: Multiple options
   - Mean: h_N = mean({h_j : j ∈ N(i)})
   - Max: h_N = max({h_j : j ∈ N(i)})  (element-wise)
   - LSTM: Order neighbors, apply LSTM

3. UPDATE: Concatenate and transform
   h_i' = σ(W × [h_i || h_N])

GraphSAGE is inductive:
  - Can generalize to new nodes
  - Doesn't need to retrain on new graph
""")

def graphsage_layer(A, H, W_self, W_neigh, sample_size=None):
    """GraphSAGE layer with mean aggregation"""
    n_nodes = H.shape[0]
    H_new = []
    
    for i in range(n_nodes):
        # Get neighbors
        neighbors = np.where(A[i] > 0)[0]
        
        # Sample if specified
        if sample_size and len(neighbors) > sample_size:
            neighbors = np.random.choice(neighbors, sample_size, replace=False)
        
        # Aggregate (mean)
        if len(neighbors) > 0:
            h_neigh = np.mean(H[neighbors], axis=0)
        else:
            h_neigh = np.zeros(H.shape[1])
        
        # Concatenate and transform
        h_concat = np.concatenate([H[i], h_neigh])
        h_new = np.tanh(W_self @ H[i] + W_neigh @ h_neigh)
        H_new.append(h_new)
    
    return np.array(H_new)

print("GraphSAGE scales to million-node graphs via sampling")
```

## Graph Attention Network (GAT)

```python
print("\n=== GRAPH ATTENTION (GAT) ===")
print("""
Learn attention weights for aggregation

Attention mechanism:
  e_ij = LeakyReLU(a^T [W h_i || W h_j])
  
  α_ij = softmax_j(e_ij) = exp(e_ij) / Σ_k exp(e_ik)

Update:
  h_i' = σ(Σ_j α_ij W h_j)

Multi-head attention (like Transformers):
  - K attention heads
  - Concatenate or average outputs
  
  h_i' = ||_{k=1}^K σ(Σ_j α_ij^k W^k h_j)

Benefits:
  - Different importance for different neighbors
  - Implicitly handles node degree
  - Multi-head captures different relationship types
""")

def gat_attention(H_i, H_j, W, a):
    """Compute attention coefficient"""
    Wh_i = W @ H_i
    Wh_j = W @ H_j
    concat = np.concatenate([Wh_i, Wh_j])
    e = np.maximum(0.2 * (a @ concat), a @ concat)  # LeakyReLU
    return e

print("GAT learns WHICH neighbors to pay attention to")
```

## GNN Tasks

```python
print("\n=== GRAPH LEARNING TASKS ===")
print("""
1. NODE CLASSIFICATION:
   Predict label for each node
   - Social network: User interests
   - Citation: Paper topic
   
   Use node embeddings → classifier

2. LINK PREDICTION:
   Predict if edge should exist
   - Friend recommendation
   - Knowledge graph completion
   
   score(i, j) = f(h_i, h_j)  # dot product, MLP, etc.

3. GRAPH CLASSIFICATION:
   Predict label for entire graph
   - Molecule property prediction
   - Protein function
   
   Need graph-level representation:
   h_G = READOUT({h_i : i ∈ V})
   - Mean/sum/max pooling
   - Attention-based pooling

4. GRAPH GENERATION:
   Generate new graphs
   - Drug design
   - Material discovery
""")
```

## Key Points

- **Graphs**: Represent relational data (nodes + edges)
- **Message passing**: Aggregate neighbor information
- **GCN**: Spectral convolution, normalized aggregation
- **GraphSAGE**: Sampling for scalability, inductive
- **GAT**: Attention-weighted aggregation
- **Tasks**: Node, link, graph classification

## Reflection Questions

1. Why can't standard CNNs be applied directly to graphs?
2. How does the number of GNN layers affect what a node can "see"?
3. When would you choose GAT over GCN?
