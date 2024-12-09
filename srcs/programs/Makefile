DATA_DIR = ./data
DEPO_DIR = ./depo
PRED_DIR = ./predictions

clean:
	@rm -f $(DATA_DIR)/*.csv 
	@rm -f $(DEPO_DIR)/*.npy
	@rm -f $(PRED_DIR)/*.csv 
	@echo "\033[91mDeleted all CSV files in the data folder.\033[0m"
	@echo "\033[91mDeleted all NPY files in the depo folder.\033[0m"

create:
	@mkdir -p $(DATA_DIR)
	@mkdir -p $(DEPO_DIR)
	@mkdir -p $(PRED_DIR)
	@echo "\033[92mCreated data, depo and predictions folders.\033[0m"