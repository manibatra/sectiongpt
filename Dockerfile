FROM python:3.9-slim

WORKDIR /app

COPY requirements.txt .

# Install the dependencies and clean cache
RUN apt-get update && apt-get install -y --no-install-recommends gcc g++ \
    && pip install --no-cache-dir -r requirements.txt \
    && apt-get remove --purge -y gcc g++ \
    && apt-get autoremove -y \
    && rm -rf /var/lib/apt/lists/*

COPY . .

CMD ["python", "app.py"]

# Expose the required port
EXPOSE 80