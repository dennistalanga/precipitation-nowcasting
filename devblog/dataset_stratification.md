# 📖 DevBlog: Meteorological Dataset Stratification via K-Means Clustering

**Author:** Dennis Bob Talanga  
**Category:** Data-Centric AI & Weather Analytics  
**Context:** Project *Precipitation Radar Nowcasting using Deep Learning*

---

## The Engineering Challenge: Validation Distribution Shift
An initial chronological split of the radar archives produced an unexpected training artifact: the validation and evaluation losses were consistently lower than the training metrics. 

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


## Stratified Continuous Split Search

Because the current pipeline implementation loads data via continuous, non-overlapping datetime ranges, the framework handles timeline partitioning by searching for contiguous multi-month blocks rather than shuffling individual 10-day archives. This continuous bounding strategy ensures total chronological sequence integrity within each isolated split.

To handle this continuous loading layout, an automated search engine inside `notebooks/03_dataset_difficulty_analysis.ipynb` scans the entire timeline to find continuous windows that match on weather difficulty. The script evaluates the search space and provides two split setup suggestions:

1. **Minimal Difficulty Difference:** This option picks the windows with the mathematically smallest difference in cluster averages, making the splits as similar as possible in overall difficulty.
2. **Distribution-Constrained Match:** This option adds a strict rule requiring each split (train, val, test) to contain at least one archive from every cluster. This ensures that the splits do not just match on a generic average, but share a highly similar weather distribution that includes rare convective storms in every phase.


```text
Target Allocation Strategy:
[Complete 15-Month Archive Timeline] 
        │
        ▼ [Combinatorial Window Search Engine]
        ├──► Training Split (9 Months Continuous)   ──► Cluster Mix: [55.6% Low, 29.6% Med, 14.8% High]  (15 / 8 / 4 Archives)
        ├──► Validation Split (3 Months Continuous) ──► Cluster Mix: [55.6% Low, 33.3% Med, 11.1% High]  (5 / 3 / 1 Archives)
        └──► Testing Split (3 Months Continuous)    ──► Cluster Mix: [55.6% Low, 22.2% Med, 22.2% High]  (5 / 2 / 2 Archives)

```

### Ordering Regimes & Breaking Search Parity: The Composite Difficulty Score

To bridge the gap between discrete categorical cluster IDs and raw continuous data tracking, the pipeline computes a continuous **Composite Difficulty Score** for each archive block. This score aggregates the standardized features via weighted configurations:

```python
df["computed_difficulty_score"] = (
    (df["temporal_variability"] / (df["temporal_variability"].max() + 1e-6)) * weights.get("temporal_variability", 0.25) +
    (df["spatial_variance"] / (df["spatial_variance"].max() + 1e-6)) * weights.get("spatial_variance", 0.20) +
    (df["rain_coverage_heavy"] / (df["rain_coverage_heavy"].max() + 1e-6)) * weights.get("heavy_tail", 0.20) +
    (df["rain_coverage_light"] / (df["rain_coverage_light"].max() + 1e-6)) * weights.get("frequency", 0.15) +
    (df["max_intensity"] / (df["max_intensity"].max() + 1e-6)) * weights.get("max_intensity", 0.10) +
    (df["average_motion"] / (df["average_motion"].max() + 1e-6)) * weights.get("dynamics", 0.10)
)
```

This composite metric serves two critical algorithmic functions within the data engineering track:

1. **Monotonic Cluster ID Sorting:** Raw K-Means assignments default to arbitrary cluster indexing. The pipeline groups archives by their cluster ID, computes the true mean difficulty score for each group, and dynamically maps the IDs so that Cluster 0, 1, and 2 are systematically sorted from lowest to highest meteorological difficulty.
2. **Combinatorial Search Tie-Breaking:** When searching for optimal timeline splits, different multi-month windows often resolve to identical discrete cluster distributions. The search engine resolves this parity by computing a secondary continuous metric tracking the average difficulty error (`total_difficulty_error`) across candidate train, validation, and test windows. Sorting the results by `["mae_score", "difficulty_error"]` uses the continuous score as a rigorous tie-breaker, guaranteeing optimal statistical alignment.

---

### The Analytical Objective
By aligning the distribution mix of weather regimes across the splits as closely as possible, the seasonal bias is reduced significantly. This statistical balancing ensures that changes in validation or test metrics are far more likely to reflect genuine architectural or optimization improvements rather than random variations in seasonal weather difficulty.

