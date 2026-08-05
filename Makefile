.PHONY: preprocess stratify train search evaluate plot_grid plot_gif plot_history plot_search clean

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

clean:
	rm -rf src/__pycache__ src/*/__pycache__ src/*/*/__pycache__
