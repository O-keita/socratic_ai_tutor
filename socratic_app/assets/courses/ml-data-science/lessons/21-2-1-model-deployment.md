# Model Deployment Strategies

## Introduction

Model deployment is the process of making trained ML models available to serve predictions in production. This lesson covers deployment patterns, serving architectures, and best practices.

## Deployment Patterns

```python
import numpy as np
import pandas as pd

print("=== DEPLOYMENT PATTERNS ===")
print("""
1. BATCH INFERENCE:
   - Run predictions on scheduled intervals
   - Process large datasets at once
   - Results stored for later use
   - Good for: Reports, recommendations, offline scoring

2. REAL-TIME INFERENCE:
   - Predictions on-demand via API
   - Low latency requirements
   - Single or small batches
   - Good for: Search, fraud detection, chatbots

3. STREAMING INFERENCE:
   - Continuous prediction on data streams
   - Event-driven processing
   - Near real-time
   - Good for: IoT, monitoring, real-time recommendations

4. EMBEDDED INFERENCE:
   - Model runs on edge device
   - No network dependency
   - Privacy-preserving
   - Good for: Mobile apps, IoT devices, offline use
""")
```

## Model Serving Architectures

```python
print("\n=== SERVING ARCHITECTURES ===")
print("""
1. SIMPLE WEB SERVICE:
   ┌──────────────┐
   │   Flask/     │
   │   FastAPI    │ ← HTTP Request (features)
   │    +         │ → HTTP Response (prediction)
   │   Model      │
   └──────────────┘
   
   Pros: Simple, quick to deploy
   Cons: Limited scalability

2. MICROSERVICE ARCHITECTURE:
   ┌──────────┐     ┌──────────────┐
   │ Load     │────→│ Model Server │──→ Model 1
   │ Balancer │     │ (replicated) │──→ Model 2
   └──────────┘     └──────────────┘    ...
   
   Pros: Scalable, fault-tolerant
   Cons: More complex

3. MODEL SERVER (TFServing, TorchServe):
   ┌──────────────────┐
   │  Model Server    │
   │  ┌────────────┐  │
   │  │ Model v1   │  │
   │  │ Model v2   │  │ ← gRPC/REST
   │  │ Model v3   │  │
   │  └────────────┘  │
   └──────────────────┘
   
   Pros: Optimized, versioning built-in
   Cons: Framework-specific

4. SERVERLESS (Lambda, Cloud Functions):
   Request → Function (loads model) → Response
   
   Pros: Auto-scaling, pay-per-use
   Cons: Cold start latency, size limits
""")
```

## FastAPI Model Serving

```python
print("\n=== FASTAPI SERVING EXAMPLE ===")

fastapi_example = '''
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import numpy as np

app = FastAPI(title="ML Model API", version="1.0")

# Load model at startup
model = None

@app.on_event("startup")
async def load_model():
    global model
    model = joblib.load("model.pkl")
    print("Model loaded successfully")

class PredictionRequest(BaseModel):
    features: list[float]
    
class PredictionResponse(BaseModel):
    prediction: int
    probability: float
    model_version: str

@app.post("/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest):
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    # Convert to array
    X = np.array(request.features).reshape(1, -1)
    
    # Get prediction
    prediction = int(model.predict(X)[0])
    probability = float(model.predict_proba(X)[0].max())
    
    return PredictionResponse(
        prediction=prediction,
        probability=probability,
        model_version="1.0.0"
    )

@app.get("/health")
async def health_check():
    return {"status": "healthy", "model_loaded": model is not None}

# Run with: uvicorn app:app --host 0.0.0.0 --port 8000
'''

print(fastapi_example)
```

## Containerization with Docker

```python
print("\n=== DOCKER DEPLOYMENT ===")

dockerfile_example = '''
# Dockerfile for ML model serving

FROM python:3.9-slim

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy model and code
COPY model.pkl .
COPY app.py .

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s \\
    CMD curl -f http://localhost:8000/health || exit 1

# Run server
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
'''

print("Dockerfile:")
print(dockerfile_example)

docker_compose = '''
# docker-compose.yml
version: "3.8"
services:
  model-server:
    build: .
    ports:
      - "8000:8000"
    environment:
      - MODEL_PATH=/app/model.pkl
    volumes:
      - ./models:/app/models
    deploy:
      replicas: 3
      resources:
        limits:
          memory: 2G
'''

print("\nDocker Compose:")
print(docker_compose)
```

## Model Versioning and A/B Testing

```python
print("\n=== A/B TESTING MODELS ===")
print("""
A/B Testing: Compare model performance in production

SETUP:
  - Model A (control): Current production model
  - Model B (treatment): New challenger model
  - Traffic split: e.g., 90% A, 10% B

METRICS TO TRACK:
  - Business metrics (conversion, revenue)
  - Model metrics (accuracy, latency)
  - User experience (errors, timeouts)

TRAFFIC ROUTING:

Option 1: Random assignment
  if random() < 0.1:
      return model_b.predict(x)
  return model_a.predict(x)

Option 2: Sticky assignment (by user)
  if hash(user_id) % 100 < 10:
      return model_b.predict(x)
  return model_a.predict(x)

Option 3: Feature flags (LaunchDarkly, etc.)
  if feature_flag.is_enabled("new_model", user):
      return model_b.predict(x)
  return model_a.predict(x)
""")

def ab_test_router(user_id, models, traffic_split={'A': 0.9, 'B': 0.1}):
    """Route user to model based on traffic split"""
    # Deterministic assignment based on user_id
    bucket = hash(user_id) % 100
    
    cumulative = 0
    for model_name, fraction in traffic_split.items():
        cumulative += fraction * 100
        if bucket < cumulative:
            return model_name, models[model_name]
    
    return list(models.keys())[-1], list(models.values())[-1]

# Demo
class MockModel:
    def __init__(self, name):
        self.name = name
    def predict(self, x):
        return f"Prediction from {self.name}"

models = {'A': MockModel('A'), 'B': MockModel('B')}

print("\nA/B test routing for sample users:")
for user_id in ['user_001', 'user_002', 'user_003', 'user_123', 'user_456']:
    model_name, model = ab_test_router(user_id, models)
    print(f"  {user_id} → Model {model_name}")
```

## Canary Deployments

```python
print("\n=== CANARY DEPLOYMENT ===")
print("""
Gradually roll out new model to catch issues early

PHASES:
1. Deploy to canary (1-5% traffic)
2. Monitor metrics closely
3. If healthy, increase traffic (10%, 25%, 50%, 100%)
4. If issues, rollback immediately

ROLLBACK TRIGGERS:
  - Error rate spike
  - Latency increase
  - Metric degradation
  - Anomaly detection alerts

Example rollout schedule:
  Hour 0:   1% traffic to new model
  Hour 1:   5% (if healthy)
  Hour 4:  10%
  Hour 8:  25%
  Hour 24: 50%
  Hour 48: 100%

SHADOW MODE (alternative):
  - Run new model on all traffic
  - Log predictions but don't serve
  - Compare to production model
  - Deploy when confident
""")

class CanaryDeployer:
    def __init__(self, old_model, new_model):
        self.old_model = old_model
        self.new_model = new_model
        self.canary_percentage = 0
        self.error_counts = {'old': 0, 'new': 0}
        self.request_counts = {'old': 0, 'new': 0}
    
    def set_canary_percentage(self, percentage):
        self.canary_percentage = percentage
        print(f"Canary percentage set to {percentage}%")
    
    def predict(self, x):
        use_canary = np.random.random() < (self.canary_percentage / 100)
        
        try:
            if use_canary:
                self.request_counts['new'] += 1
                return self.new_model.predict(x)
            else:
                self.request_counts['old'] += 1
                return self.old_model.predict(x)
        except Exception as e:
            if use_canary:
                self.error_counts['new'] += 1
            else:
                self.error_counts['old'] += 1
            raise e
    
    def get_error_rates(self):
        rates = {}
        for model in ['old', 'new']:
            if self.request_counts[model] > 0:
                rates[model] = self.error_counts[model] / self.request_counts[model]
            else:
                rates[model] = 0
        return rates

print("Canary deployment pattern implemented")
```

## Kubernetes Deployment

```python
print("\n=== KUBERNETES DEPLOYMENT ===")

k8s_deployment = '''
# model-deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: model-server
  labels:
    app: model-server
spec:
  replicas: 3
  selector:
    matchLabels:
      app: model-server
  template:
    metadata:
      labels:
        app: model-server
    spec:
      containers:
      - name: model-server
        image: my-registry/model-server:v1.0
        ports:
        - containerPort: 8000
        resources:
          requests:
            memory: "1Gi"
            cpu: "500m"
          limits:
            memory: "2Gi"
            cpu: "1000m"
        readinessProbe:
          httpGet:
            path: /health
            port: 8000
          initialDelaySeconds: 10
          periodSeconds: 5
        livenessProbe:
          httpGet:
            path: /health
            port: 8000
          initialDelaySeconds: 30
          periodSeconds: 10
---
apiVersion: v1
kind: Service
metadata:
  name: model-service
spec:
  selector:
    app: model-server
  ports:
    - protocol: TCP
      port: 80
      targetPort: 8000
  type: LoadBalancer
'''

print(k8s_deployment)
```

## Key Points

- **Deployment patterns**: Batch, real-time, streaming, embedded
- **Architecture**: Web service, microservice, model server, serverless
- **Containerization**: Docker for reproducible deployments
- **A/B testing**: Compare models in production
- **Canary deployment**: Gradual rollout with monitoring
- **Kubernetes**: Scalable, production-grade orchestration

## Reflection Questions

1. When would you choose batch inference over real-time?
2. What are the trade-offs between A/B testing and shadow mode?
3. How do you determine if a canary deployment is "healthy"?
