FROM python:3.13-slim
WORKDIR /app
RUN pip install poetry==1.8.4
COPY pyproject.toml ./
RUN poetry config virtualenvs.create false && poetry install --only main --no-root
COPY agent_customer_support ./agent_customer_support
EXPOSE 8800
CMD ["uvicorn", "agent_customer_support.server:app", "--host", "0.0.0.0", "--port", "8800"]
