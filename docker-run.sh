#!/bin/bash

# Docker Compose runner script for RAG-ES with Streamlit
# This script helps you run the entire stack in Docker

set -e

echo "🚀 RAG-ES Docker Compose Runner"
echo "================================"

# Check if .env file exists
if [ ! -f ".env" ]; then
    echo "❌ .env file not found!"
    echo "Please copy env.example to .env and configure your settings:"
    echo "  cp env.example .env"
    echo "  # Edit .env and add your OpenAI API key"
    exit 1
fi

# Check if Docker is running
if ! docker info > /dev/null 2>&1; then
    echo "❌ Docker is not running!"
    echo "Please start Docker Desktop and try again."
    exit 1
fi

echo "✅ Docker is running"
echo "✅ .env file found"

# Function to show help
show_help() {
    echo ""
    echo "Usage: $0 [COMMAND]"
    echo ""
    echo "Commands:"
    echo "  up        Start all services (default)"
    echo "  down      Stop all services"
    echo "  build     Build/rebuild all images"
    echo "  logs      Show logs for all services"
    echo "  logs-api  Show logs for API service only"
    echo "  logs-ui   Show logs for Streamlit UI only"
    echo "  restart   Restart all services"
    echo "  clean     Stop and remove all containers, networks, and volumes"
    echo "  help      Show this help message"
    echo ""
    echo "Services will be available at:"
    echo "  - Streamlit UI: http://localhost:8501"
    echo "  - FastAPI API: http://localhost:8000"
    echo "  - Elasticsearch: http://localhost:9200"
    echo "  - Kibana: http://localhost:5601"
    echo ""
}

# Function to start services
start_services() {
    echo "🔨 Building and starting services..."
    docker-compose up --build -d
    
    echo ""
    echo "⏳ Waiting for services to be ready..."
    echo "This may take a few minutes on first run..."
    
    # Wait for services to be healthy
    echo "Waiting for Elasticsearch..."
    docker-compose exec elasticsearch sh -c 'until curl -f http://localhost:9200/_cluster/health; do sleep 5; done'
    
    echo "Waiting for API..."
    docker-compose exec api sh -c 'until curl -f http://localhost:8000/health; do sleep 5; done'
    
    echo "Waiting for Streamlit..."
    docker-compose exec streamlit sh -c 'until curl -f http://localhost:8501/_stcore/health; do sleep 5; done'
    
    echo ""
    echo "🎉 All services are ready!"
    echo ""
    echo "📱 Access your applications:"
    echo "  - Chat UI: http://localhost:8501"
    echo "  - API Docs: http://localhost:8000/docs"
    echo "  - Elasticsearch: http://localhost:9200"
    echo "  - Kibana: http://localhost:5601"
    echo ""
    echo "📊 To view logs: $0 logs"
    echo "🛑 To stop: $0 down"
}

# Function to show logs
show_logs() {
    if [ "$1" = "api" ]; then
        docker-compose logs -f api
    elif [ "$1" = "ui" ]; then
        docker-compose logs -f streamlit
    else
        docker-compose logs -f
    fi
}

# Main script logic
case "${1:-up}" in
    up)
        start_services
        ;;
    down)
        echo "🛑 Stopping all services..."
        docker-compose down
        ;;
    build)
        echo "🔨 Building all images..."
        docker-compose build --no-cache
        ;;
    logs)
        show_logs
        ;;
    logs-api)
        show_logs api
        ;;
    logs-ui)
        show_logs ui
        ;;
    restart)
        echo "🔄 Restarting all services..."
        docker-compose restart
        ;;
    clean)
        echo "🧹 Cleaning up all containers, networks, and volumes..."
        docker-compose down -v --remove-orphans
        docker system prune -f
        ;;
    help|--help|-h)
        show_help
        ;;
    *)
        echo "❌ Unknown command: $1"
        show_help
        exit 1
        ;;
esac
