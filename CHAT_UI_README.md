# RAG Chat UI

A simple Python-based chat interface for the RAG Document Ingestion System using Streamlit.

## Features

- 🤖 **Interactive Chat Interface**: Clean, modern chat UI with message history
- 📚 **Document Queries**: Ask questions about literature and places in books
- 🌤️ **Weather Information**: Get current weather conditions for any location
- 🗺️ **Travel Planning**: Combine literature + weather for trip planning
- 🛡️ **Smart Routing**: Automatically routes queries to appropriate handlers
- 📊 **System Status**: Real-time health monitoring of API and agent
- 💡 **Example Queries**: Pre-built example questions to get started

## Quick Start

### 1. Prerequisites

Make sure you have the RAG system running:

```bash
# Start the API server (in one terminal)
python run.py

# Or manually:
python -m app.main
```

### 2. Start the Chat UI

```bash
# Easy way - use the launcher script
python start_chat.py

# Or directly with Streamlit
streamlit run chat_ui.py
```

The chat interface will open in your browser at `http://localhost:8501`

## Usage

### Example Queries

The sidebar includes example queries you can click to try:

- **Document Questions**: "What places did Mark Twain visit?"
- **Weather Queries**: "What's the weather in Rome?"
- **Combined Queries**: "I want to visit places Twain went to in Italy - what's the weather?"
- **Character Questions**: "Tell me about the characters in the book"

### Features

1. **Real-time Status**: The sidebar shows if the API server and agent are healthy
2. **Message History**: All conversations are preserved during the session
3. **Source Information**: Click to expand sources, weather data, and locations
4. **Clear Chat**: Use the "Clear Chat" button to start fresh
5. **Processing Info**: See which route was taken and processing time

### Error Handling

The UI gracefully handles:
- API server not running
- Agent errors
- Network timeouts
- Invalid responses

## Technical Details

### Architecture

```
Streamlit UI → FastAPI Server → LangGraph Agent → Elasticsearch/Weather APIs
```

### Files

- `chat_ui.py`: Main Streamlit application
- `start_chat.py`: Launcher script with dependency checking
- `requirements.txt`: Updated with Streamlit dependency

### Dependencies

- `streamlit>=1.28.0`: Web UI framework
- `requests`: HTTP client for API calls
- All existing RAG system dependencies

## Troubleshooting

### Common Issues

1. **"API Server: Unhealthy"**
   - Make sure the FastAPI server is running on port 8000
   - Check if Elasticsearch is running: `docker-compose up -d`

2. **"Agent: Unhealthy"**
   - Check the API logs for agent initialization errors
   - Verify OpenAI API key is configured in `.env`

3. **"Cannot connect to the API"**
   - Ensure the API server is running: `python run.py`
   - Check if port 8000 is available

4. **Streamlit not found**
   - Install Streamlit: `pip install streamlit>=1.28.0`
   - Or use the launcher script: `python start_chat.py`

### Debug Mode

To run Streamlit in debug mode:

```bash
streamlit run chat_ui.py --logger.level debug
```

## Customization

### Styling

The UI uses custom CSS for styling. You can modify the styles in the `st.markdown()` section of `chat_ui.py`.

### API Configuration

To change the API endpoint, modify the `API_BASE_URL` variable in `chat_ui.py`:

```python
API_BASE_URL = "http://localhost:8000"  # Change this if needed
```

### Port Configuration

To change the Streamlit port, modify the launcher script or use:

```bash
streamlit run chat_ui.py --server.port 8502
```

## Development

### Adding New Features

1. Modify `chat_ui.py` for UI changes
2. Update the API endpoints in `app/routers/agent.py` for backend changes
3. Test with the launcher script

### Testing

1. Start the API server: `python run.py`
2. Start the chat UI: `python start_chat.py`
3. Test various query types in the browser

## Support

For issues with the RAG system itself, check the main README.md.
For UI-specific issues, check the Streamlit documentation: https://docs.streamlit.io/
