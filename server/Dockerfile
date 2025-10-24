# Use an official lightweight Python image
FROM python:3.12-slim

# Set environment variables for better performance
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
        PORT=8080

        # Set working directory
        WORKDIR /app

        # Install system dependencies
        RUN apt-get update && apt-get install -y --no-install-recommends \
            build-essential \
                curl && \
                    rm -rf /var/lib/apt/lists/*

                    # Copy dependency files
                    COPY requirements.txt .

                    # Install Python dependencies
                    RUN pip install --no-cache-dir --upgrade pip && \
                        pip install --no-cache-dir -r requirements.txt

                        # Copy your FastAPI app
                        COPY app.py .
                        COPY v2.py .

                        # Expose the port (Cloud Run uses PORT env variable)
                        EXPOSE 8080

                        # Use shell form to allow environment variable interpolation
                        CMD gunicorn app:app -k uvicorn.workers.UvicornWorker \
                            --workers 4 --bind 0.0.0.0:$PORT --timeout 120 --log-level info