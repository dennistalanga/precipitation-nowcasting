## Detailed Project Structure

```text
precipitation-nowcasting/
├── assets/             # Documentation visuals and diagrams
├── configs/            # YAML configuration files for make commands
├── data/               # Local dataset files (git-ignored)
│   ├── difficulty/     # Dataset difficulty assessment results
│   ├── processed/      # Preprocessed uncompressed radar data
│   └── raw/            # Compressed radar data from MeteoNet
├── deployment/         # 🚀 Production Serving Infrastructure Layer
│   ├── app.py          # FastAPI web service endpoint mapping and setup logic
│   ├── pipeline.py     # Live matrix preprocessing and validation bridge
│   ├── schemas.py      # Strict runtime Pydantic response data schemas
│   └── test_api.py     # Endpoint automated integration test execution suite
├── environment.yml     # Conda environment config file
├── Dockerfile          # Multi-stage container instruction blueprint
├── .dockerignore       # Context block firewall filter parameters
├── Makefile            # Complete pipeline automation orchestration interface
├── devblog/            # 📖 Technical engineering articles and design journals
├── notebooks/          # Exploration, prototyping, and evaluation
├── output/             # Experiment artifacts (git-ignored)
│   ├── logs/           # Local execution logs
│   ├── models/         # Model checkpoints, experiment info, and train history
│   ├── evaluation/     # Test predictions, computed metrics, and plots
│   └── tuning/         # Random search results and performance graphs
├── README.md
└── src/
    ├── data/           # Preprocessing, datasets, stratification
    ├── evaluation/     # Model evaluation pipeline, loss- and metrics definitions
    ├── models/         # Model implementations and factories
    ├── inference/      # Model inference and live predictions (not yet implemented)
    ├── training/       # Model training pipeline
    ├── tuning/         # Hyperparameter optimization
    ├── utils/          # Shared utilities and logging
    └── visualization/  # Plotting for training, prediction, and evaluation
```

## Full Tech Stack

| Category                   | Technologies                  | Use Case
| -------------------------- | ----------------------------- | ----------------------------------------------------------------------------------------------------------
| **Language**               | Python 3.11                   | Core application runtime and pipeline implementation
| **Deep Learning**          | PyTorch                       | Neural network implementation, tensor operations, automatic differentiation, and GPU training
| **Web Framework**          | FastAPI                       | Production-grade model serving microservice and routing layer                                              |
| **Data Validation**        | Pydantic                      | Strict runtime configuration parsing, type enforcement, and response JSON serialization                    |
| **Server Engine**          | Uvicorn                       | High-performance, asynchronous ASGI web server processing                                                  |
| **System Validation**      | Requests                      | Automated end-to-end endpoint infrastructure testing                                                       |
| **Data Analysis**          | Scikit-learn                  | Unsupervised K-Means clustering for archive difficulty stratification and balanced dataset construction
| **Numerical Computing**    | NumPy                         | Numerical processing, array manipulation, preprocessing, memory-mapped data loading, and evaluation metrics
| **Data Processing**        | Pandas                        | Archive characterization, difficulty scoring, clustering results, and experiment summaries
| **Computer Vision**        | OpenCV, ImageIO, Scikit-Image | Radar image preprocessing, resizing, validation, and animated prediction visualizations 
| **Data Loading**           | PyTorch Dataset & DataLoader  | Lazy-loading spatiotemporal radar sequences with parallel data streaming
| **Visualization**          | Matplotlib, Seaborn, Cartopy  | Training diagnostics, prediction analysis, statistical visualizations, and geospatial plotting
| **Experiment Tracking**    | JSON, CSV                     | Lightweight experiment metadata, metrics, configuration, and reproducibility
| **Interactive Analysis**   | Jupyter Notebook              | Exploratory data analysis, preprocessing design, prediction inspection, and experimental prototyping
| **Environment Management** | Conda                         | Dependency management and reproducible development environments
| **Automation**             | Bash, Make                    | Pipeline automation for preprocessing, training, evaluation, hyperparameter search, and visualization
| **Version Control**        | Git                           | Distributed source control and project version management