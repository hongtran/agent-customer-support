.PHONY: build run test lint eval eval-retrieval infra-up infra-down

build:
	poetry install
run:
	poetry run uvicorn agent_customer_support.server:app --reload --port 8800 --env-file .env
test:
	poetry run pytest -v
lint:
	poetry run ruff format agent_customer_support tests && poetry run ruff check --fix agent_customer_support tests && poetry run mypy agent_customer_support
eval:
	poetry run python -m eval.run_eval --mode both
eval-retrieval:
	poetry run python -m eval.run_eval --mode retrieval
infra-up:
	docker compose up -d
infra-down:
	docker compose down
