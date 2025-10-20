#!/usr/bin/env python3
"""Container startup script for the RAG Document Ingestion System."""

import os
import sys
import time
import requests

def check_elasticsearch():
    """Check if Elasticsearch is running."""
    try:
        # In Docker container, Elasticsearch is available at the service name
        elasticsearch_url = os.getenv("ELASTICSEARCH_URL", "http://elasticsearch:9200")
        response = requests.get(f"{elasticsearch_url}/_cluster/health", timeout=5)
        return response.status_code == 200
    except:
        return False

def wait_for_elasticsearch():
    """Wait for Elasticsearch to be ready."""
    print("Waiting for Elasticsearch to be ready...")
    for i in range(60):  # Wait up to 60 seconds
        if check_elasticsearch():
            print("Elasticsearch is ready!")
            return True
        print(f"Waiting for Elasticsearch... ({i+1}/60)")
        time.sleep(1)
    print("Elasticsearch failed to start within 60 seconds")
    return False

def main():
    """Main startup function for container environment."""
    print("RAG Document Ingestion System - Container Startup")
    print("=" * 50)
    
    # Check if .env file exists
    if not os.path.exists(".env"):
        print("ERROR: .env file not found!")
        print("Please copy env.example to .env and configure your settings:")
        print("  cp env.example .env")
        sys.exit(1)
    
    # Check if OpenAI API key is configured
    try:
        with open(".env", "r") as f:
            env_content = f.read()
            if "your_openai_api_key_here" in env_content:
                print("ERROR: OpenAI API key not configured!")
                print("Please edit .env file and replace 'your_openai_api_key_here' with your actual OpenAI API key.")
                print("You can get your API key from: https://platform.openai.com/api-keys")
                sys.exit(1)
    except Exception as e:
        print(f"Error reading .env file: {e}")
        sys.exit(1)
    
    # Wait for Elasticsearch to be ready
    if not wait_for_elasticsearch():
        print("ERROR: Elasticsearch is not available!")
        sys.exit(1)
    
    # Start the FastAPI application
    print("\nStarting FastAPI application...")
    print("API will be available at: http://localhost:8000")
    print("API documentation at: http://localhost:8000/docs")
    
    try:
        import uvicorn
        from app.config import settings
        
        uvicorn.run(
            "app.main:app",
            host="0.0.0.0",  # Listen on all interfaces in container
            port=8000,
            reload=False,    # Disable reload in production container
            log_level="info"
        )
    except KeyboardInterrupt:
        print("\nShutting down...")
    except Exception as e:
        print(f"Error starting application: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
