DATA_DIR = ./data
DEPO_DIR = ./depo
PRED_DIR = ./predictions

clean:
	@rm -fr $(DATA_DIR)
	@rm -fr $(DEPO_DIR)
	@rm -fr $(PRED_DIR)
	@find .. -type d -name "__pycache__" -exec rm -rf {} +
	@echo "\033[91mDeleted all CSV files in the data folder.\033[0m"
	@echo "\033[91mDeleted all NPY files in the depo folder.\033[0m"
	@echo "\033[91mDeleted all __pycache__ directories.\033[0m"

create:
	@mkdir -p $(DATA_DIR)
	@mkdir -p $(DEPO_DIR)
	@mkdir -p $(PRED_DIR)
	@echo "\033[92mCreated data, depo and predictions folders.\033[0m"


virtual:
	@if [ ! -d "myenv" ]; then \
		python3.10 -m venv myenv && \
		. myenv/bin/activate && pip install -r ../../requirements.txt && \
		echo "\033[92mCreated virtual environment.\033[0m"; \
	else \
		echo "\033[93mVirtual environment already exists.\033[0m"; \
	fi

install:
	pip install -r ../../requirements.txt
	@echo "\033[92mInstalled all python packages\033[0m"