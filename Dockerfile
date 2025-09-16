FROM python:3.11.9-slim
WORKDIR /app   
# Install system dependencies needed for scientific Python packages
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    gcc \
    g++ \
    libatlas-base-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy the Pip files into our working derectory 
COPY ["Pipfile", "Pipfile.lock", "./"]
# Install pipenv + dependencies into system
RUN pip install --no-cache-dir pipenv \
    && pipenv install --deploy --system

# Copy the rest of the project files
COPY . /app

# We need to expose the 9696 port because we're not able to communicate with Docker outside it
EXPOSE 8000
# If we run the Docker image, we want our churn app to be running
ENTRYPOINT ["waitress-serve", "--listen=0.0.0.0:8000", "predict:app"]




