# 📖 DevBlog: Meteorological Dataset Stratification via K-Means Clustering

**Author:** Dennis Bob Talanga  
**Category:** Data-Centric AI & Weather Analytics  
**Context:** Project *Precipitation Radar Nowcasting using Deep Learning*

---

## The Engineering Challenge: Validation Distribution Shift
An initial chronological split of the radar archives produced a highly unexpected training artifact: the validation loss and evaluation metrics were consistently higher than the training metrics. 

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

---

## Stratified Continuous Split Search

Because short-term spatiotemporal models require consecutive frames to capture velocity vectors, the dataset loader must operate on continuous datetime ranges. It cannot simply shuffle individual 10-day archives randomly, as this would break temporal continuity at boundary edges.

To bridge this constraint, an automated split recommendation engine inside `notebooks/03_dataset_difficulty_analysis.ipynb` executes a windowed combinatorial search. It scans the continuous timeline to extract windows whose overall distribution of the three K-Means clusters matches as closely as possible.

```text
Target Allocation Strategy:
[Complete 3-Year Archive Timeline] 
        │
        ▼ [Combinatorial Window Search Engine]
        ├──► Training Split (6 Months Continuous)   ──► Cluster Mix: [55% Low, 30% Med, 15% High]
        ├──► Validation Split (2 Months Continuous) ──► Cluster Mix: [54% Low, 31% Med, 15% High]
        └──► Testing Split (2 Months Continuous)    ──► Cluster Mix: [55% Low, 30% Med, 15% High]
```

### The Analytical Result
By enforcing statistical parity across the splits, changes in validation or testing performance are mathematically guaranteed to reflect genuine architectural improvements rather than random changes in seasonal weather difficulty.
