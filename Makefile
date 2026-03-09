.PHONY: install test lint run-baseline run-mc

install:
	pip install -r requirements.txt --break-system-packages

test:
	pytest tests/ -v --tb=short

test-cov:
	pytest tests/ --cov=src --cov-report=term-missing

run-baseline:
	python -m experiments.baseline_strategy_eval

run-mc:
	python -m experiments.monte_carlo_race_test

run-rl:
	python -m experiments.rl_training_experiment

lint:
	python -m flake8 src/ --max-line-length=100

