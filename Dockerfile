# syntax=docker/dockerfile:1
FROM python:3.12-slim-bookworm


# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    TOKENIZERS_PARALLELISM=false


# Copy requirements.txt first for better Docker cache
COPY requirements.txt ./

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy project files
ADD . /app
WORKDIR /app

# Create directories for persistence
RUN mkdir -p /data/db /data/uploads

# Expose port
EXPOSE 8000



# Set environment variables for persistence
ENV DB_PATH=/data/db/chatbot.db \
    UPLOAD_FOLDER=/data/uploads


# Entrypoint
CMD ["python", "run.py"]
