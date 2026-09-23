# Use Python 3.11
FROM python:3.11-slim

# Set working directory
WORKDIR /main

# Prevent Python from creating .pyc files
ENV PYTHONDONTWRITEBYTECODE=1

# Prevent Python output buffering
ENV PYTHONUNBUFFERED=1

# Copy requirements first for better Docker caching
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy application files
COPY main.py .
COPY docs_files ./docs_files

# Streamlit default port
EXPOSE 8501

# Start Streamlit
CMD ["streamlit", "run", "main.py", "--server.address=0.0.0.0", "--server.port=8501"]