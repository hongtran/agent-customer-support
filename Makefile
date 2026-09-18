.PHONY: build run test lint eval eval-retrieval eval-triage eval-guardrail infra-up infra-down modal-download modal-deploy

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
eval-triage:
	poetry run python -m eval.triage_eval
eval-guardrail:
	poetry run python -m eval.guardrail_eval
infra-up:
	docker compose up -d
infra-down:
	docker compose down
# Self-hosted Qwen on Modal (needs the `modal` CLI: uv tool install modal && modal setup)
modal-download:
	modal run qwen/download_model.py
modal-deploy:
	modal deploy qwen/serve.py
