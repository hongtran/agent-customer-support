FROM python:3.13-slim
WORKDIR /app
# Same major version as the one that writes poetry.lock (lock-version 2.1).
RUN pip install poetry==2.4.1
# The lock is copied so prod installs exactly the tested versions. Without it every
# build re-resolved the ^ ranges, which once pulled a qdrant-client too new for the
# Qdrant server.
COPY pyproject.toml poetry.lock ./
RUN poetry config virtualenvs.create false && poetry install --only main --no-root
COPY agent_customer_support ./agent_customer_support
EXPOSE 8080
CMD ["uvicorn", "agent_customer_support.server:app", "--host", "0.0.0.0", "--port", "8080"]
