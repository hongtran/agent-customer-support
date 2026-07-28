.PHONY: build run test lint infra-up infra-down

build:
	poetry install
run:
	poetry run uvicorn agent_customer_support.server:app --reload --port 8800 --env-file .env
test:
	poetry run pytest -v
lint:
	poetry run ruff format agent_customer_support tests && poetry run ruff check --fix agent_customer_support tests && poetry run mypy agent_customer_support
infra-up:
	docker compose up -d
infra-down:
	docker compose down
