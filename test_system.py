#!/usr/bin/env python3
"""Test script for the RAG Document Ingestion System."""

import requests
import json
import time
import os

BASE_URL = "http://localhost:8000"

def test_health():
    """Test system health."""
    print("Testing system health...")
    try:
        response = requests.get(f"{BASE_URL}/health")
        if response.status_code == 200:
            print("✓ System is healthy")
            print(f"  Elasticsearch status: {response.json().get('elasticsearch', {}).get('status', 'unknown')}")
            return True
        else:
            print(f"✗ Health check failed: {response.status_code}")
            return False
    except Exception as e:
        print(f"✗ Health check error: {e}")
        return False

def test_upload():
    """Test document upload."""
    print("\nTesting document upload...")
    
    # Check if sample book exists
    if not os.path.exists("sample_book.txt"):
        print("✗ Sample book file not found")
        return None
    
    try:
        with open("sample_book.txt", "rb") as f:
            files = {"file": ("sample_book.txt", f, "text/plain")}
            response = requests.post(f"{BASE_URL}/api/ingest/upload", files=files)
        
        if response.status_code == 200:
            data = response.json()
            document_id = data["document_id"]
            print(f"✓ Document uploaded successfully")
            print(f"  Document ID: {document_id}")
            return document_id
        else:
            print(f"✗ Upload failed: {response.status_code}")
            print(f"  Response: {response.text}")
            return None
    except Exception as e:
        print(f"✗ Upload error: {e}")
        return None

def test_processing_status(document_id):
    """Test processing status."""
    print(f"\nTesting processing status for document {document_id}...")
    
    max_wait = 60  # Wait up to 60 seconds
    wait_time = 0
    
    while wait_time < max_wait:
        try:
            response = requests.get(f"{BASE_URL}/api/ingest/status/{document_id}")
            if response.status_code == 200:
                data = response.json()
                status = data["status"]
                message = data["message"]
                progress = data.get("progress", 0)
                
                print(f"  Status: {status} ({progress}%) - {message}")
                
                if status == "completed":
                    print("✓ Document processing completed")
                    return True
                elif status == "failed":
                    print("✗ Document processing failed")
                    return False
                
                time.sleep(2)
                wait_time += 2
            else:
                print(f"✗ Status check failed: {response.status_code}")
                return False
        except Exception as e:
            print(f"✗ Status check error: {e}")
            return False
    
    print("✗ Processing timeout")
    return False

def test_search():
    """Test search functionality."""
    print("\nTesting search functionality...")
    
    test_queries = [
        "What is machine learning?",
        "types of learning algorithms",
        "neural networks and deep learning"
    ]
    
    for query in test_queries:
        print(f"\n  Testing query: '{query}'")
        try:
            payload = {
                "query": query,
                "top_k": 5,
                "rerank": True,
                "compression": True
            }
            
            response = requests.post(f"{BASE_URL}/api/search/search", json=payload)
            
            if response.status_code == 200:
                data = response.json()
                results = data["results"]
                processing_time = data["processing_time_ms"]
                
                print(f"    ✓ Found {len(results)} results in {processing_time:.2f}ms")
                
                # Show first result
                if results:
                    first_result = results[0]
                    print(f"    Best match: {first_result['text'][:100]}...")
                    print(f"    Score: {first_result['score']:.3f}")
                    print(f"    Chapter: {first_result['chapter_info']['title']}")
            else:
                print(f"    ✗ Search failed: {response.status_code}")
                print(f"    Response: {response.text}")
        except Exception as e:
            print(f"    ✗ Search error: {e}")

def test_documents_list():
    """Test documents listing."""
    print("\nTesting documents list...")
    try:
        response = requests.get(f"{BASE_URL}/api/ingest/documents")
        if response.status_code == 200:
            data = response.json()
            documents = data["documents"]
            print(f"✓ Found {len(documents)} indexed documents")
            
            for doc in documents:
                print(f"  - {doc['title']} (ID: {doc['document_id']})")
        else:
            print(f"✗ Documents list failed: {response.status_code}")
    except Exception as e:
        print(f"✗ Documents list error: {e}")

def main():
    """Run all tests."""
    print("RAG Document Ingestion System - Test Suite")
    print("=" * 50)
    
    # Test 1: Health check
    if not test_health():
        print("\nSystem is not healthy. Please check Elasticsearch and API.")
        return
    
    # Test 2: Upload document
    document_id = test_upload()
    if not document_id:
        print("\nUpload failed. Cannot continue with other tests.")
        return
    
    # Test 3: Processing status
    if not test_processing_status(document_id):
        print("\nProcessing failed. Cannot continue with search tests.")
        return
    
    # Test 4: Search functionality
    test_search()
    
    # Test 5: Documents list
    test_documents_list()
    
    print("\n" + "=" * 50)
    print("Test suite completed!")
    print(f"API documentation: {BASE_URL}/docs")

if __name__ == "__main__":
    main()
