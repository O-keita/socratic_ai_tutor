# Visualization Principles

## Introduction

Effective data visualization communicates complex information clearly and efficiently. This lesson covers fundamental principles for creating impactful visualizations.

## The Grammar of Graphics

```python
import numpy as np
import pandas as pd

print("=== GRAMMAR OF GRAPHICS ===")
print("""
Components of a Visualization:

1. DATA
   - The underlying dataset
   - Variables being visualized

2. AESTHETICS (aes)
   - How data maps to visual properties
   - Position (x, y)
   - Color, size, shape, line type

3. GEOMETRIES (geom)
   - How data is represented
   - Points, lines, bars, areas

4. FACETS
   - Subplots by category
   - Small multiples

5. STATISTICS
   - Data transformations
   - Binning, smoothing, aggregation

6. COORDINATES
   - Coordinate system
   - Cartesian, polar, map projections

7. THEMES
   - Non-data elements
   - Labels, legends, fonts
""")
```

## Choosing the Right Chart

```python
print("\n=== CHOOSING THE RIGHT CHART ===")
print("""
BY DATA TYPE AND PURPOSE:

DISTRIBUTION (one variable):
  - Histogram: Binned frequency distribution
  - Density plot: Smooth distribution estimate
  - Box plot: Summary statistics
  - Violin plot: Distribution + density

RELATIONSHIP (two+ variables):
  - Scatter plot: Two continuous variables
  - Line plot: Continuous over ordered dimension (time)
  - Bubble chart: Three variables (x, y, size)
  - Heatmap: Two categorical + one continuous

COMPARISON:
  - Bar chart: Categories vs. values
  - Grouped bar: Categories across groups
  - Stacked bar: Part-to-whole by category
  - Dot plot: Alternative to bar chart

COMPOSITION:
  - Pie chart: Part-to-whole (use sparingly!)
  - Stacked area: Composition over time
  - Treemap: Hierarchical composition

TREND:
  - Line chart: Change over time
  - Area chart: Magnitude over time
  - Slope chart: Change between two points
""")

# Decision tree for chart selection
print("\n=== QUICK SELECTION GUIDE ===")
print("""
Ask yourself:

1. How many variables?
   - One: Distribution chart
   - Two: Relationship or comparison chart
   - Three+: May need faceting or color/size encoding

2. What are the data types?
   - Categorical + Categorical: Heatmap, grouped bar
   - Categorical + Numeric: Bar chart, box plot
   - Numeric + Numeric: Scatter plot, line chart

3. What's the goal?
   - Show distribution: Histogram, box plot
   - Compare groups: Bar chart, box plot
   - Show relationship: Scatter plot
   - Show trend: Line chart
   - Show composition: Stacked bar, pie (rarely)
""")
```

## Principles of Good Design

```python
print("\n=== DESIGN PRINCIPLES ===")
print("""
1. MAXIMIZE DATA-INK RATIO
   - Show data, minimize decoration
   - Remove unnecessary gridlines, borders
   - Avoid 3D effects, shadows
   
   "Above all else, show the data" - Edward Tufte

2. AVOID CHARTJUNK
   - No unnecessary decoration
   - No misleading embellishments
   - No distracting backgrounds

3. MAINTAIN INTEGRITY
   - Start y-axis at zero for bar charts
   - Don't truncate axes to exaggerate differences
   - Keep aspect ratios appropriate

4. USE COLOR PURPOSEFULLY
   - Sequential: Low to high values
   - Diverging: Values around a midpoint
   - Categorical: Distinct categories
   - Consider colorblind accessibility

5. LABEL CLEARLY
   - Descriptive titles
   - Axis labels with units
   - Legends when needed
   - Direct labeling when possible

6. SIMPLIFY
   - One main message per chart
   - Remove clutter
   - Guide the viewer's eye
""")
```

## Color Usage

```python
print("\n=== EFFECTIVE COLOR USE ===")
print("""
COLOR SCALES:

1. SEQUENTIAL (ordered data)
   - Low to high values
   - Single hue, varying lightness
   - Example: Light blue → Dark blue
   
2. DIVERGING (midpoint matters)
   - Values above/below a center
   - Two hues meeting at neutral
   - Example: Blue ← Gray → Red

3. CATEGORICAL (distinct groups)
   - Maximally different colors
   - No implied order
   - Example: Blue, Orange, Green, Red

COLOR ACCESSIBILITY:
  - 8% of men are colorblind
  - Avoid red-green distinctions alone
  - Use patterns/shapes as backup
  - Test with colorblindness simulators

COMMON MISTAKES:
  ✗ Too many colors (cognitive overload)
  ✗ Rainbow color scales (misleading)
  ✗ 3D pie charts (distorted perception)
  ✗ Similar colors for different categories
""")

# Color palette examples
print("\nRecommended Palettes:")
print("""
Sequential: Blues, Greens, Grays
Diverging: RdBu (Red-Blue), BrBG (Brown-Teal)
Categorical: Set2, Paired, Tab10
Colorblind-safe: viridis, cividis, Okabe-Ito
""")
```

## Avoiding Common Mistakes

```python
print("\n=== COMMON VISUALIZATION MISTAKES ===")
print("""
1. PIE CHARTS FOR COMPARISON
   ✗ Hard to compare slice sizes
   ✓ Use bar charts instead

2. TRUNCATED Y-AXIS
   ✗ Makes small differences look huge
   ✓ Start at zero for bar charts

3. DUAL Y-AXES
   ✗ Easy to mislead, hard to interpret
   ✓ Use facets or normalize data

4. TOO MANY CATEGORIES
   ✗ Rainbow explosion of colors
   ✓ Group small categories as "Other"
   ✓ Focus on top categories

5. 3D CHARTS
   ✗ Perspective distorts perception
   ✓ Stick to 2D

6. OVER-DECORATION
   ✗ Excessive gridlines, borders, shadows
   ✓ Minimize non-data elements

7. MISLEADING SCALES
   ✗ Inconsistent intervals
   ✗ Non-zero baseline for area/bar
   ✓ Honest, consistent scales

8. UNLABELED CHARTS
   ✗ "What are the axes?"
   ✓ Clear titles, labels, units
""")
```

## Creating Effective Annotations

```python
print("\n=== ANNOTATIONS AND LABELS ===")
print("""
TITLES:
  - Be descriptive, not just names
  - ✗ "Sales Data"
  - ✓ "Sales increased 25% after campaign launch"

AXIS LABELS:
  - Include units: "Revenue ($M)", "Time (hours)"
  - Make orientation readable

ANNOTATIONS:
  - Highlight key findings
  - Add context for anomalies
  - Point out trends or patterns

LEGENDS:
  - Place close to data
  - Use direct labeling when possible
  - Order meaningfully

DATA LABELS:
  - Add values to bars when few categories
  - Don't crowd the visualization
  - Round appropriately
""")

# Example annotation strategy
print("\nAnnotation Checklist:")
print("""
□ Clear, informative title
□ Subtitle with context (if needed)
□ Labeled axes with units
□ Legend (if colors/shapes used)
□ Data source attribution
□ Key points annotated
□ Appropriate number format
""")
```

## Designing for Different Audiences

```python
print("\n=== AUDIENCE CONSIDERATIONS ===")
print("""
EXECUTIVE/STAKEHOLDER:
  - Simple, high-level view
  - Clear takeaway message
  - Minimal technical detail
  - Emphasize business impact

DATA ANALYST/SCIENTIST:
  - More detail acceptable
  - Statistical information
  - Interactive exploration
  - Technical annotations

GENERAL PUBLIC:
  - Very simple
  - Familiar chart types
  - Clear explanations
  - Avoid jargon

PRESENTATION vs DOCUMENT:
  - Presentation: Larger fonts, simpler
  - Document: More detail, smaller

ASK YOURSELF:
  - What question does this answer?
  - What action should result?
  - What's the one main message?
""")
```

## Key Points

- **Grammar of Graphics**: Data, aesthetics, geometries, facets, coordinates, themes
- **Choose wisely**: Match chart type to data type and purpose
- **Maximize data-ink**: Remove unnecessary elements
- **Use color purposefully**: Sequential, diverging, or categorical
- **Maintain integrity**: Honest scales, proper baselines
- **Label clearly**: Titles, axes, legends
- **Know your audience**: Tailor complexity and detail

## Reflection Questions

1. Why are pie charts often criticized, and when might they be appropriate?
2. How does the choice of color palette affect accessibility?
3. What makes a visualization "lie" about the data?
