#!/usr/bin/env python3
"""
Simple Streamlit-based chat UI for the RAG Document Ingestion System.

This provides a user-friendly interface to interact with the LangGraph agent
that can handle document queries, weather information, and combined queries.
"""

import streamlit as st
import requests
import json
import time
from datetime import datetime
from typing import Dict, Any, List

# Configuration
import os
API_BASE_URL = os.getenv("API_BASE_URL", "http://localhost:8000")
AGENT_ENDPOINT = f"{API_BASE_URL}/api/agent"

def check_api_health() -> bool:
    """Check if the API is running and healthy."""
    try:
        response = requests.get(f"{API_BASE_URL}/health", timeout=5)
        return response.status_code == 200
    except:
        return False

def check_agent_health() -> Dict[str, Any]:
    """Check if the agent is healthy."""
    try:
        response = requests.get(f"{AGENT_ENDPOINT}/health", timeout=10)
        if response.status_code == 200:
            return response.json()
        else:
            return {"status": "unhealthy", "error": f"HTTP {response.status_code}"}
    except Exception as e:
        return {"status": "unhealthy", "error": str(e)}

def send_query(query: str) -> Dict[str, Any]:
    """Send a query to the agent API."""
    try:
        payload = {"query": query}
        response = requests.post(
            f"{AGENT_ENDPOINT}/query",
            json=payload,
            timeout=30
        )
        
        if response.status_code == 200:
            result = response.json()
            return result
        else:
            return {
                "error": f"API Error: {response.status_code}",
                "details": response.text
            }
    except requests.exceptions.Timeout:
        return {"error": "Request timed out. The query might be too complex."}
    except requests.exceptions.ConnectionError:
        return {"error": "Cannot connect to the API. Make sure the server is running."}
    except Exception as e:
        return {"error": f"Unexpected error: {str(e)}"}

def format_sources(sources: List[Dict]) -> str:
    """Format sources for display."""
    if not sources:
        return "No sources available"
    
    formatted = []
    for i, source in enumerate(sources, 1):
        if source.get('type') == 'document':
            title = source.get('title', 'Unknown Document')
            section = source.get('section', 'Unknown Section')
            formatted.append(f"{i}. **{title}** - {section}")
        elif source.get('type') == 'weather':
            location = source.get('location', 'Unknown Location')
            formatted.append(f"{i}. **Weather Data** - {location}")
        else:
            formatted.append(f"{i}. {source.get('title', 'Unknown Source')}")
    
    return "\n".join(formatted)

def format_weather_data(weather_data: List[Dict]) -> str:
    """Format weather data for display."""
    if not weather_data:
        return "No weather data available"
    
    formatted = []
    for weather in weather_data:
        city = weather.get('city', 'Unknown')
        temp_data = weather.get('temperature', {})
        conditions = weather.get('conditions', {})
        
        # Extract temperature and condition info
        current_temp = temp_data.get('current', 'N/A') if isinstance(temp_data, dict) else 'N/A'
        description = conditions.get('description', 'N/A') if isinstance(conditions, dict) else 'N/A'
        humidity = weather.get('humidity', 'N/A')
        
        formatted.append(f"**{city}**: {current_temp}°C, {description}, Humidity: {humidity}%")
    
    return "\n".join(formatted)

def main():
    """Main Streamlit application."""
    st.set_page_config(
        page_title="GoRAGo",
        page_icon="🗺️",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Custom CSS for better styling
    st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .chat-message {
        padding: 1rem;
        border-radius: 10px;
        margin-bottom: 1rem;
        border-left: 4px solid #1f77b4;
    }
    .user-message {
        background-color: #f0f2f6;
        border-left-color: #1f77b4;
        color: #333333 !important;
    }
    .user-message strong {
        color: #1a1a1a !important;
        font-weight: 600;
    }
    .user-message * {
        color: #333333 !important;
    }
    .assistant-message {
        background-color: #e8f4fd;
        border-left-color: #28a745;
        color: #333333;
    }
    .assistant-message strong {
        color: #1a1a1a;
        font-weight: 600;
    }
    .user-message p {
        color: #333333;
        margin: 0.5rem 0;
    }
    .assistant-message p {
        color: #333333;
        margin: 0.5rem 0;
    }
    .error-message {
        background-color: #f8d7da;
        border-left-color: #dc3545;
        color: #721c24;
    }
    .source-info {
        background-color: #d1ecf1;
        border-left-color: #17a2b8;
        padding: 0.5rem;
        margin-top: 0.5rem;
        border-radius: 5px;
        font-size: 0.9rem;
    }
    .status-healthy {
        color: #28a745;
        font-weight: bold;
    }
    .status-unhealthy {
        color: #dc3545;
        font-weight: bold;
    }
    /* Ensure expandable content is visible */
    .streamlit-expanderContent {
        color: #333333;
    }
    .streamlit-expanderContent strong {
        color: #1a1a1a;
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Header
    st.markdown('<h1 class="main-header">🗺️ GoRAGo</h1>', unsafe_allow_html=True)
    st.markdown("**AI-powered system that can:**")
    st.markdown("• Answer questions about locations, experiences, and insights from \"The Innocents Abroad\"")
    st.markdown("• Provide current weather information for destinations you're considering")
    st.markdown("• Intelligently combine both sources when needed to give comprehensive travel advice")
    
    # Add spacing to move chat down
    st.markdown("<br><br>", unsafe_allow_html=True)
    
    # Sidebar for system status and info
    with st.sidebar:
        st.header("System Status")
        
        # Check API health
        api_healthy = check_api_health()
        if api_healthy:
            st.markdown('<p class="status-healthy">✅ API Server: Healthy</p>', unsafe_allow_html=True)
        else:
            st.markdown('<p class="status-unhealthy">❌ API Server: Unhealthy</p>', unsafe_allow_html=True)
            st.error("Make sure the FastAPI server is running on http://localhost:8000")
            st.stop()
        
        # Check agent health
        agent_health = check_agent_health()
        if agent_health.get("status") == "healthy":
            st.markdown('<p class="status-healthy">✅ Agent: Healthy</p>', unsafe_allow_html=True)
        else:
            st.markdown('<p class="status-unhealthy">❌ Agent: Unhealthy</p>', unsafe_allow_html=True)
            st.error(f"Agent error: {agent_health.get('error', 'Unknown error')}")
        
        st.divider()
        
        # Agent capabilities
        st.header("Capabilities")
        st.markdown("""
        - 📚 **Literature Insights**: Ask about locations and experiences from "The Innocents Abroad"
        - 🌤️ **Weather Information**: Get current weather conditions for any destination
        - 🗺️ **Travel Planning**: Combine literature insights + weather for comprehensive travel advice
        - 🛡️ **Smart Routing**: Automatically routes your questions to the right service
        """)
    
    # Initialize session state
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    if "user_input" not in st.session_state:
        st.session_state.user_input = ""
    
    if "last_input" not in st.session_state:
        st.session_state.last_input = ""
    
    # Chat interface
    col1, col2 = st.columns([4, 1])
    
    with col1:
        user_input = st.text_input(
            "Ask me anything about documents or weather:",
            value=st.session_state.user_input,
            placeholder="e.g., What places did Mark Twain visit?",
            key="main_input",
            on_change=None
        )
    
    with col2:
        send_button = st.button("Send", type="primary", use_container_width=True)
    
    # Clear session state input after processing
    if st.session_state.user_input:
        st.session_state.user_input = ""
    
    # Process query (either button click or Enter key)
    if (send_button and user_input) or (user_input and user_input != st.session_state.last_input and user_input.strip()):
        # Update last input to prevent duplicate processing
        st.session_state.last_input = user_input
        
        # Add user message
        st.session_state.messages.append({
            "role": "user",
            "content": user_input,
            "timestamp": datetime.now()
        })
        
        # Show loading spinner
        with st.spinner("Thinking..."):
            response = send_query(user_input)
        
        # Add assistant response
        st.session_state.messages.append({
            "role": "assistant",
            "content": response,
            "timestamp": datetime.now()
        })
        
        # Clear input
        st.rerun()
    
    # Display chat messages grouped by conversation turns (most recent first)
    # Group messages into conversation pairs
    conversation_turns = []
    i = 0
    while i < len(st.session_state.messages):
        if i + 1 < len(st.session_state.messages) and st.session_state.messages[i]["role"] == "user" and st.session_state.messages[i + 1]["role"] == "assistant":
            # Found a user-assistant pair
            conversation_turns.append({
                "user": st.session_state.messages[i],
                "assistant": st.session_state.messages[i + 1]
            })
            i += 2
        else:
            # Handle orphaned messages
            if st.session_state.messages[i]["role"] == "user":
                conversation_turns.append({
                    "user": st.session_state.messages[i],
                    "assistant": None
                })
            i += 1
    
    # Display conversation turns (most recent first)
    for turn in reversed(conversation_turns):
        # User question
        st.markdown(f"""
        <div class="chat-message user-message">
            <strong>You:</strong> {turn["user"]["content"]}
        </div>
        """, unsafe_allow_html=True)
        
        # Assistant response (if available)
        if turn["assistant"]:
            response = turn["assistant"]["content"]
            
            if response.get("error") is not None:
                st.markdown(f"""
                <div class="chat-message error-message">
                    <strong>Assistant:</strong> {response["error"]}
                </div>
                """, unsafe_allow_html=True)
            else:
                # Main response
                answer = response.get("answer", "")
                
                if not answer or answer == "None" or answer is None:
                    answer = "I apologize, but I couldn't generate a proper response. This might be due to insufficient context or an internal error. Please try rephrasing your question or check if the system is working properly."
                
                st.markdown(f"""
                <div class="chat-message assistant-message">
                    <strong>Assistant:</strong> {answer}
                </div>
                """, unsafe_allow_html=True)
                
                # Route taken
                if response.get("route_taken"):
                    st.info(f"**Route taken:** {response['route_taken']}")
                
                # Processing time
                if response.get("processing_time_ms"):
                    st.info(f"**Processing time:** {response['processing_time_ms']:.2f}ms")
                
                # Sources
                if response.get("sources"):
                    with st.expander("📚 Sources", expanded=False):
                        st.markdown(format_sources(response["sources"]))
                
                # Weather data
                if response.get("weather_data"):
                    with st.expander("🌤️ Weather Data", expanded=False):
                        st.markdown(format_weather_data(response["weather_data"]))
                
                # Locations
                if response.get("locations"):
                    with st.expander("📍 Locations Mentioned", expanded=False):
                        # Extract location names from LocationInfo objects
                        location_names = []
                        for location in response["locations"]:
                            if isinstance(location, dict):
                                location_names.append(location.get("name", "Unknown"))
                            else:
                                location_names.append(str(location))
                        locations = ", ".join(location_names)
                        st.markdown(f"**Locations:** {locations}")
        
        # Add separator between conversation turns
        st.markdown("---")
    
    # Clear chat button
    if st.session_state.messages:
        if st.button("🗑️ Clear Chat", type="secondary"):
            st.session_state.messages = []
            st.rerun()
    
    # Footer
    st.divider()
    st.markdown("""
    <div style="text-align: center; color: #666; font-size: 0.9rem;">
        GoRAGo - Powered by LangGraph Agent with Elasticsearch and OpenAI
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
