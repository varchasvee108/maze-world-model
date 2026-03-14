.PHONY: setup train infer test-unit test-integration test-all clean generate-data help

setup:
	pip install -e .
	pip install -e ".[dev]"

generate-data:
	python -m data.processor

train:
	python -m scripts.train --config config/base.toml

infer:
	@if [ -z "$(IMG)" ] || [ -z "$(CKPT)" ]; then \
		echo "Usage: make infer IMG=path/to/img.png CKPT=path/to/model.ckpt"; \
	else \
		python -m scripts.infer --image_path $(IMG) --checkpoint_path $(CKPT); \
	fi

test-unit:
	pytest tests -m "not integration and not slow"

test-slow:
	pytest tests -m "slow"

test-integration:
	pytest tests -m "integration"

test-all:
	pytest tests

	