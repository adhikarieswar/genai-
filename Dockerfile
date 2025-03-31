# Use the official Python 3.9 slim image as the base
FROM python:3.9-slim

# Set the working directory inside the container
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy the requirements file into the container
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code
COPY . .

# Expose port 8501 for Streamlit
EXPOSE 8505

# Health check to ensure the app is running
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD curl --fail http://localhost:8505/_stcore/health || exit 1


# Command to run the Streamlit app
ENTRYPOINT ["streamlit", "run", "chatboatproject.py", "--server.port=8505", "--server.address=0.0.0.0"]