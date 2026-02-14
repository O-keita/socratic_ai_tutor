# Object Detection

## Introduction

Object detection goes beyond classification to localize and identify multiple objects within an image. It combines classification (what) with localization (where).

## Object Detection vs Classification

```python
import numpy as np
import pandas as pd

print("=== OBJECT DETECTION ===")
print("""
CLASSIFICATION: Single label for whole image
  "This image contains a cat"

LOCALIZATION: Single object + bounding box
  "Cat at (x=100, y=50, w=200, h=150)"

DETECTION: Multiple objects + boxes
  "Cat at (100, 50, 200, 150)"
  "Dog at (300, 80, 180, 200)"
  "Person at (450, 20, 100, 280)"

Bounding box format:
  (x, y, width, height) or
  (x_min, y_min, x_max, y_max)
""")
```

## Key Concepts

```python
print("\n=== KEY CONCEPTS ===")
print("""
1. BOUNDING BOX
   Rectangle around object
   4 values: position + size

2. CONFIDENCE SCORE
   How sure the model is (0-1)
   Filter low-confidence detections

3. CLASS LABEL
   What object is detected
   Multiple classes possible

4. INTERSECTION OVER UNION (IoU)
   Measures overlap between boxes
   IoU = Area of Overlap / Area of Union
""")

def calculate_iou(box1, box2):
    """Calculate IoU between two boxes [x1, y1, x2, y2]"""
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    
    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    
    union = area1 + area2 - intersection
    
    return intersection / union if union > 0 else 0

# Example
box1 = [100, 100, 200, 200]  # Ground truth
box2 = [120, 110, 220, 210]  # Prediction

iou = calculate_iou(box1, box2)
print(f"\nIoU example:")
print(f"  Box 1: {box1}")
print(f"  Box 2: {box2}")
print(f"  IoU: {iou:.3f}")
print(f"\n  IoU > 0.5 typically means 'correct detection'")
```

## Two-Stage Detectors (R-CNN Family)

```python
print("\n=== TWO-STAGE DETECTORS ===")
print("""
Stage 1: REGION PROPOSAL
  - Generate candidate regions
  - "Where might objects be?"

Stage 2: CLASSIFICATION + REFINEMENT
  - Classify each region
  - Refine bounding box

R-CNN Evolution:
  R-CNN → Fast R-CNN → Faster R-CNN

R-CNN (2014):
  - Selective Search for ~2000 regions
  - CNN on each region separately
  - Very slow (47s per image)

Fast R-CNN (2015):
  - CNN on whole image once
  - RoI pooling to extract region features
  - 0.5s per image

Faster R-CNN (2016):
  - Region Proposal Network (RPN)
  - End-to-end trainable
  - 0.2s per image
""")

print("""
Faster R-CNN architecture:

Image → Backbone CNN → Feature Map
                          ↓
                   Region Proposal Network (RPN)
                          ↓
                   Proposed Regions (~300)
                          ↓
                   RoI Pooling/Align
                          ↓
                   Classification + Box Regression
                          ↓
                   Final Detections
""")
```

## One-Stage Detectors (YOLO, SSD)

```python
print("\n=== ONE-STAGE DETECTORS ===")
print("""
Single pass through network:
  - Divide image into grid
  - Predict boxes + classes at each cell
  - Much faster than two-stage

YOLO (You Only Look Once):
  - Image divided into S×S grid
  - Each cell predicts B boxes + C class probs
  - End-to-end, real-time

SSD (Single Shot Detector):
  - Multi-scale feature maps
  - Anchors at different scales
  - Better for small objects

Speed comparison:
  Faster R-CNN: ~7 FPS
  SSD: ~59 FPS
  YOLOv3: ~30 FPS
  YOLOv5: ~140 FPS
""")

print("""
YOLO approach:

Image (448×448) → CNN → 7×7×30 tensor

7×7: Grid cells
30: Per cell predictions
  - 2 bounding boxes × 5 values (x, y, w, h, conf)
  - 20 class probabilities

Total: 7×7×2 = 98 candidate boxes
After NMS: Final detections
""")
```

## Non-Maximum Suppression (NMS)

```python
print("\n=== NON-MAXIMUM SUPPRESSION ===")
print("""
PROBLEM: Multiple overlapping detections for same object

SOLUTION: NMS keeps only the best box

Algorithm:
1. Sort boxes by confidence score
2. Take highest confidence box
3. Remove all boxes with IoU > threshold with selected box
4. Repeat until no boxes left

Threshold typically 0.5
""")

def nms(boxes, scores, iou_threshold=0.5):
    """Simple NMS implementation"""
    indices = np.argsort(scores)[::-1]
    keep = []
    
    while len(indices) > 0:
        current = indices[0]
        keep.append(current)
        
        if len(indices) == 1:
            break
        
        remaining = indices[1:]
        ious = np.array([calculate_iou(boxes[current], boxes[i]) for i in remaining])
        
        # Keep boxes with low IoU
        indices = remaining[ious < iou_threshold]
    
    return keep

# Example
boxes = [
    [100, 100, 200, 200],
    [110, 105, 205, 195],
    [300, 300, 400, 400],
    [105, 100, 210, 205]
]
scores = [0.9, 0.75, 0.8, 0.85]

kept = nms(boxes, scores, 0.5)
print(f"Boxes: {len(boxes)} → After NMS: {len(kept)}")
print(f"Kept indices: {kept}")
```

## Anchor Boxes

```python
print("\n=== ANCHOR BOXES ===")
print("""
Pre-defined box shapes at each location:

Why anchors?
  - Objects have different aspect ratios
  - Cars: wide, People: tall
  - Each anchor handles specific shape

Common anchors:
  - 1:1 (square)
  - 1:2 (tall)
  - 2:1 (wide)
  
At multiple scales for different object sizes.

Model predicts OFFSETS from anchors:
  - Δx, Δy: Center offset
  - Δw, Δh: Size scaling
  
Final box = anchor + predicted offsets
""")

def apply_offsets(anchor, offsets):
    """Apply predicted offsets to anchor box"""
    ax, ay, aw, ah = anchor
    dx, dy, dw, dh = offsets
    
    x = ax + dx * aw
    y = ay + dy * ah
    w = aw * np.exp(dw)
    h = ah * np.exp(dh)
    
    return [x, y, w, h]

anchor = [100, 100, 50, 80]  # Center x, y, width, height
offsets = [0.1, -0.2, 0.3, 0.1]

result = apply_offsets(anchor, offsets)
print(f"Anchor: {anchor}")
print(f"Offsets: {offsets}")
print(f"Final box: {[round(v, 1) for v in result]}")
```

## Evaluation Metrics

```python
print("\n=== EVALUATION METRICS ===")
print("""
Mean Average Precision (mAP):

1. For each class:
   - Sort detections by confidence
   - For each detection:
     - Match to ground truth if IoU > threshold
     - Mark as TP or FP
   - Calculate precision-recall curve
   - AP = Area under PR curve

2. mAP = mean of APs across all classes

Common thresholds:
  - mAP@0.5: IoU threshold 0.5
  - mAP@0.75: Stricter threshold
  - mAP@[0.5:0.95]: Average over multiple thresholds

COCO metrics use mAP@[0.5:0.95] by default.
""")

print("""
Example calculation:

Detections (sorted by conf):
  1. Conf=0.95, IoU=0.8 → TP
  2. Conf=0.90, IoU=0.3 → FP
  3. Conf=0.85, IoU=0.7 → TP
  4. Conf=0.80, IoU=0.2 → FP

Precision at each point:
  P@1: 1/1 = 1.0
  P@2: 1/2 = 0.5
  P@3: 2/3 = 0.67
  P@4: 2/4 = 0.5

AP approximated by area under PR curve.
""")
```

## Modern Architectures

```python
print("\n=== MODERN ARCHITECTURES ===")
print("""
YOLOv5/v8:
  - State-of-the-art speed
  - Easy to use (PyTorch)
  - Multiple sizes (n, s, m, l, x)

Detectron2 (Facebook):
  - Faster R-CNN, Mask R-CNN
  - High accuracy
  - Research-focused

EfficientDet:
  - Compound scaling
  - BiFPN feature fusion
  - Good accuracy/speed trade-off

Transformer-based:
  - DETR: End-to-end with attention
  - No anchors, no NMS
  - Simpler but needs more data
""")
```

## Key Points

- **Object detection**: Classify AND localize multiple objects
- **IoU**: Measures bounding box overlap quality
- **Two-stage**: Region proposal → classification (accurate)
- **One-stage**: Direct prediction (fast)
- **NMS**: Remove duplicate detections
- **Anchors**: Pre-defined boxes for different shapes
- **mAP**: Standard evaluation metric

## Reflection Questions

1. Why do one-stage detectors trade accuracy for speed?
2. How do anchor boxes help detect objects of different shapes?
3. What happens if you set the NMS IoU threshold too high or too low?
