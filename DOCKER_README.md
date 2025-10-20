# Docker Setup for RAG-ES with Streamlit

This guide explains how to run the RAG-ES application with Streamlit using Docker Compose.

## Prerequisites

1. **Docker Desktop** installed and running
2. **OpenAI API Key** configured in `.env` file

## Quick Start

1. **Set up environment variables:**
   ```bash
   cp env.example .env
   # Edit .env and add your OpenAI API key
   ```

2. **Run the entire stack:**
   ```bash
   ./docker-run.sh up
   ```

3. **Access the applications:**
   - **Streamlit Chat UI**: http://localhost:8501
   - **FastAPI API**: http://localhost:8000/docs
   - **Elasticsearch**: http://localhost:9200
   - **Kibana**: http://localhost:5601

## Available Commands

The `docker-run.sh` script provides several convenient commands:

```bash
# Start all services
./docker-run.sh up

# Stop all services
./docker-run.sh down

# Build/rebuild all images
./docker-run.sh build

# View logs for all services
./docker-run.sh logs

# View logs for specific service
./docker-run.sh logs-api    # API logs only
./docker-run.sh logs-ui     # Streamlit logs only

# Restart all services
./docker-run.sh restart

# Clean up everything (containers, networks, volumes)
./docker-run.sh clean

# Show help
./docker-run.sh help
```

## Manual Docker Compose Commands

If you prefer to use Docker Compose directly:

```bash
# Start all services
docker-compose up -d

# View logs
docker-compose logs -f

# Stop services
docker-compose down

# Rebuild and start
docker-compose up --build -d
```

## Services Overview

### 1. Elasticsearch (Port 9200)
- Document storage and search engine
- Health check: `curl http://localhost:9200/_cluster/health`

### 2. Kibana (Port 5601)
- Elasticsearch management interface
- Depends on Elasticsearch

### 3. FastAPI API (Port 8000)
- Backend API service
- Handles document ingestion and search
- Depends on Elasticsearch

### 4. Streamlit UI (Port 8501)
- Chat interface for interacting with the system
- Depends on FastAPI API

## Troubleshooting

### Services won't start
1. Check if Docker Desktop is running
2. Ensure `.env` file exists with valid OpenAI API key
3. Check logs: `./docker-run.sh logs`

### Port conflicts
If you have port conflicts, you can modify the ports in `docker-compose.yml`:
```yaml
ports:
  - "8502:8501"  # Change Streamlit to port 8502
  - "8001:8000"  # Change API to port 8001
```

### Memory issues
If Elasticsearch fails to start due to memory issues:
1. Increase Docker Desktop memory allocation
2. Or modify ES_JAVA_OPTS in docker-compose.yml:
   ```yaml
   environment:
     - "ES_JAVA_OPTS=-Xms256m -Xmx256m"  # Reduce memory usage
   ```

### Clean restart
If you encounter issues, try a clean restart:
```bash
./docker-run.sh clean
./docker-run.sh up
```

## Development Mode

For development, the services are configured with volume mounts so code changes are reflected immediately:

- Code changes in the API will restart the service automatically
- Streamlit will reload when you make changes to the UI

## Production Considerations

For production deployment, consider:

1. **Security**: Enable Elasticsearch security features
2. **Environment Variables**: Use proper secret management
3. **Resource Limits**: Set appropriate CPU and memory limits
4. **Networking**: Use proper network configurations
5. **Monitoring**: Add monitoring and logging solutions

## File Structure

```
RAG-ES/
├── docker-compose.yml    # Docker Compose configuration
├── Dockerfile           # Application Docker image
├── docker-run.sh        # Helper script
├── .dockerignore        # Docker ignore file
├── .env                 # Environment variables (create from env.example)
└── DOCKER_README.md     # This file
```
