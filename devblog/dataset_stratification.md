# 📖 DevBlog: Meteorological Dataset Stratification via K-Means Clustering

**Author:** Dennis Bob Talanga  
**Category:** Data-Centric AI & Weather Analytics  
**Context:** Project *Precipitation Radar Nowcasting using Deep Learning*

---

## The Engineering Challenge: Validation Distribution Shift
An initial chronological split of the radar archives produced an unexpected training artifact: the validation loss and evaluation metrics were consistently higher than the training metrics. 

A deep-dive data diagnostic revealed that this discrepancy was not a sign of extraordinary generalization capability, but rather a direct consequence of a **meteorological validation distribution shift**. Because weather activity is highly seasonal and volatile, the arbitrarily selected training period contained substantially more complex and intense precipitation patterns than the validation split. 

Evaluating models on an asymmetric, "easier" validation subset introduces significant experimental bias: architectural iterations may appear to perform better simply because the target weather distribution requires lower predictive complexity.

---

## The Solution: 9-Dimensional Feature Characterization
To build a mathematically reliable benchmark, an automated GPU-accelerated pipeline was engineered to profile every uncompressed 10-day data block before assigning it to a dataset split (`src/data/characterize_archives.py`). 

The pipeline projects each radar archive into a **9-dimensional meteorological feature descriptor vector** that quantifies the structural organization, intensity, and dynamics of the weather systems:

```text
Preprocessed Archive Block (10-Days)
├── [Coverage & Scarcity] ───► Light Rain Coverage (>0.1 mm/h)
│                             ► Moderate Rain Coverage (≥5.0 mm/h)
│                             ► Heavy Convective Tail Coverage (≥20.0 mm/h)
├── [Intensity Profiles]  ───► Mean Active Rain Intensity (mm/h)
│                             ► Maximum Peak Convective Core Intensity (mm/h)
├── [Variance & Flux]     ───► Spatial Variance (Structural organization/chaos)
│                             ► Temporal Variability (Frame-to-frame MAD)
│                             ► Rainy Frames Frequency Count
└── [Tracking Dynamics]   ───► Average System Motion (GPU Centroid Tracking)
```

---

## Unsupervised Regime Difficulty Modeling

The pipeline standardizes this 9D feature space using a `StandardScaler` to ensure uniform distance weightings and runs an **Unsupervised K-Means Clustering** engine. This mathematical layout naturally isolates three distinct physical weather regimes across the archives:

*   **Low Difficulty (Cluster 0):** Characterized by sparse, low-intensity light rain coverage, minimal temporal frame-to-frame variance, and low geometric system mobility.
*   **Medium Difficulty (Cluster 1):** Characterized by broader, continuous front-level rain bands, moderate temporal variability, and standardized tracking dynamics.
*   **High Difficulty (Cluster 2):** Characterized by severe convective activity, deep heavy-intensity tails, high spatial variance (structural storm chaos), and rapid frame-to-frame flux.


### Fusing the 9D Feature Space: The Composite Difficulty Score
To transform the discrete K-Means cluster assignments into an actionable optimization target for timeline partitioning, the pipeline calculates a continuous **Composite Difficulty Score** for each archive block. This score functions as a centralized scalar metric, mathematically fusing the standardized variance, peak convective intensity, and spatial system motion vectors. 

Instead of relying on a fragile chronological timeline, the dataset framework utilizes this composite score as its core baseline indicator. It defines the true physical complexity of the weather patterns contained within any given segment, serving as the objective valuation parameter that the continuous split search engine optimizes to match distribution profiles perfectly across the train, validation, and test splits.

---

## Stratified Continuous Split Search

Because the current pipeline implementation loads data via continuous, non-overlapping datetime ranges, the framework handles timeline partitioning by searching for contiguous multi-month blocks rather than shuffling individual 10-day archives. This continuous bounding strategy ensures total chronological sequence integrity within each isolated split.

To bridge this specific design choice, an automated split recommendation engine inside `notebooks/03_dataset_difficulty_analysis.ipynb` executes a windowed combinatorial search. It scans the continuous timeline to extract windows whose overall distribution of the three K-Means clusters matches as closely as possible.

```text
Target Allocation Strategy:
[Complete 3-Year Archive Timeline] 
        │
        ▼ [Combinatorial Window Search Engine]
        ├──► Training Split (6 Months Continuous)   ──► Cluster Mix: [55% Low, 30% Med, 15% High]
        ├──► Validation Split (2 Months Continuous) ──► Cluster Mix: [54% Low, 31% Med, 15% High]
        └──► Testing Split (2 Months Continuous)    ──► Cluster Mix: [55% Low, 30% Med, 15% High]
```

### The Analytical Objective
By aligning the distribution mix of weather regimes across the splits as closely as possible, the seasonal bias is reduced significantly. This statistical balancing ensures that changes in validation or test metrics are far more likely to reflect genuine architectural or optimization improvements rather than random variations in seasonal weather difficulty.

