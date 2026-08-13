.PHONY: preprocess stratify train search evaluate plot_grid plot_gif plot_history plot_search clean api-dev api-test docker-build docker-run

# Central environment definition
# Uses the active Conda/Conda-forge prefix if available, otherwise defaults to local
ENV    ?= $(CONDA_PREFIX)
PYTHON  = $(ENV)/bin/python

preprocess:
	$(PYTHON) -m src.data.preprocess --config configs/preprocess.yml $(OPTS)

stratify:
	$(PYTHON) -m src.data.characterize_archives --config configs/stratify.yml $(OPTS)

train:
	$(PYTHON) -m src.training.train --config configs/train_baseline.yml $(OPTS)

search:
	$(PYTHON) -m src.tuning.random_search --config configs/search.yml $(OPTS)

evaluate:
	$(PYTHON) -m src.evaluation.evaluate --config configs/evaluate.yml $(OPTS)

plot_grid:
	$(PYTHON) -m src.visualization.plot_prediction --config configs/plotting.yml $(OPTS)

plot_gif:
	$(PYTHON) -m src.visualization.create_gif --config configs/plotting.yml $(OPTS)

# Recovery command: Only needed if training was interrupted or crashed before 
# the post-training plotting step could execute automatically.
plot_history:
	$(PYTHON) -m src.visualization.plot_train_history --config configs/train_baseline.yml $(OPTS)

# Recovery command: Only needed if a random hyperparameter search sweep was 
# interrupted or crashed before the final sweep diagnostic plots could render.
plot_search:
	$(PYTHON) -m src.visualization.plot_random_search $(OPTS)

# Run the FastAPI server locally for development with hot-reloading enabled
api-dev:
	$(PYTHON) -m uvicorn deployment.app:app --reload --port 8000

# Run the automated endpoint integration verification suite
api-test:
	$(PYTHON) -m deployment.test_api

# Build the local container image
docker-build:
	docker build -t precipitation-nowcasting:latest .

# Run the production container locally
docker-run:
	docker run -p 8000:8000 -v $(PWD)/output:/workspace/output precipitation-nowcasting:latest

clean:
	rm -rf src/__pycache__ src/*/__pycache__ src/*/*/__pycache__ deployment/__pycache__
