# Swarm LLM - Emergent Multi-Agent Coordination System
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY swarm/ ./swarm/
COPY core/ ./core/
COPY config.py .
COPY run_swarm.py .

# Create output directory for logs and telemetry
RUN mkdir -p /app/output

# Set environment defaults
ENV PYTHONUNBUFFERED=1
ENV SWARM_OUTPUT_DIR=/app/output

# Default command
ENTRYPOINT ["python", "run_swarm.py"]
CMD ["--help"]
