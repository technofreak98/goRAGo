import logging
from typing import Dict, Any
from app.services.llm_service import LLMService
from app.config import settings

logger = logging.getLogger(__name__)


class GuardrailNode:
    """Guardrail node for handling out-of-scope queries."""
    
    def __init__(self):
        """Initialize guardrail node."""
        self.llm_service = LLMService()
        self.enable_guardrails = getattr(settings, 'enable_guardrails', True)
        self.oos_policy = getattr(settings, 'oos_policy', 'polite_decline')
    
    async def handle_out_of_scope(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Handle out-of-scope queries with appropriate guardrails.
        
        Args:
            state: Current agent state containing query
            
        Returns:
            Updated state with guardrail response
        """
        try:
            query = state.get("query", "")
            if not query.strip():
                return {
                    **state,
                    "final_answer": "I didn't receive a clear question. Could you please rephrase your question?",
                    "sources": []
                }
            
            logger.info(f"Handling out-of-scope query: {query}")
            
            if not self.enable_guardrails:
                return {
                    **state,
                    "final_answer": "I'm designed to help with weather information and document queries. Could you please ask about weather or documents instead?",
                    "sources": []
                }
            
            # Generate appropriate response based on policy
            if self.oos_policy == "polite_decline":
                response = await self._generate_polite_decline(query)
            else:  # safe_answer
                response = await self._generate_safe_answer(query)
            
            return {
                **state,
                "final_answer": response,
                "sources": []
            }
            
        except Exception as e:
            logger.error(f"Guardrail handling failed: {e}")
            return {
                **state,
                "final_answer": "I apologize, but I'm having trouble processing your request. I'm designed to help with weather and document queries.",
                "sources": []
            }
    
    async def _generate_polite_decline(self, query: str) -> str:
        """Generate a polite decline response."""
        try:
            system_prompt = """You are a helpful assistant that politely declines out-of-scope queries. 
            Your responses should:
            1. Be polite and respectful
            2. Briefly explain that you're designed for weather and document queries
            3. Suggest alternative topics within your scope
            4. Keep responses concise (2-3 sentences)
            
            Examples of good responses:
            - "I'm designed to help with weather information and document queries. Could you ask about the weather in a city or something from the documents I have access to?"
            - "I specialize in weather and document information. Would you like to know about weather conditions or search through available documents instead?"
            """

            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"Please politely decline this out-of-scope query: {query}"}
            ]
            
            response = await self.llm_service.chat_completion(messages)
            return response or "I'm designed to help with weather and document queries. Could you please ask about those topics instead?"
            
        except Exception as e:
            logger.error(f"Failed to generate polite decline: {e}")
            return "I'm designed to help with weather and document queries. Could you please ask about those topics instead?"
    
    async def _generate_safe_answer(self, query: str) -> str:
        """Generate a safe answer with optional tie-back."""
        try:
            system_prompt = """You are a helpful assistant that provides safe, brief answers to out-of-scope queries.
Your responses should:
1. Be safe and factual
2. Be brief (1-2 sentences)
3. Optionally tie back to weather or travel if possible
4. Not provide detailed explanations of complex topics

Examples:
- For "Explain quantum physics": "Quantum physics is a complex field of physics. I'm better suited to help with weather information or document queries."
- For "What's the capital of France": "Paris is the capital of France. I can also help with weather information for Paris or other cities."
"""

            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"Provide a safe, brief answer to this query: {query}"}
            ]
            
            response = await self.llm_service.chat_completion(messages)
            return response or "I can provide brief information, but I'm designed to help with weather and document queries."
            
        except Exception as e:
            logger.error(f"Failed to generate safe answer: {e}")
            return "I can provide brief information, but I'm designed to help with weather and document queries."


# Global guardrail node instance
guardrail_node = GuardrailNode()
