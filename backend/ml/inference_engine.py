# inference_engine.py
# Runs offline inference for Socratic AI

import json
from pathlib import Path
from typing import Optional, List, Dict
from .model_loader import model_loader
from .socratic_prompts import SocraticPromptBuilder


class InferenceEngine:
    """Handles inference for the Socratic tutoring system."""
    
    def __init__(self):
        self.config = self._load_config()
        self.prompt_builder = SocraticPromptBuilder()
        self.conversation_history: List[Dict[str, str]] = []
    
    def _load_config(self) -> dict:
        """Load configuration from config.json."""
        config_path = Path(__file__).parent.parent / 'data' / 'config.json'
        try:
            with open(config_path, 'r') as f:
                return json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            return {}
    
    def generate_response(
        self,
        user_message: str,
        difficulty: str = "intermediate",
        topic: Optional[str] = None
    ) -> str:
        """
        Generate a Socratic response to the user's message.
        
        Args:
            user_message: The user's input message
            difficulty: Difficulty level (beginner, intermediate, advanced)
            topic: Optional topic context
            
        Returns:
            The AI tutor's Socratic response
        """
        model = model_loader.get_model()
        
        if model is None:
            # Return fallback response if model isn't available
            return self._generate_fallback_response(user_message)
        
        # Build the prompt
        prompt = self.prompt_builder.build_prompt(
            user_message=user_message,
            conversation_history=self.conversation_history,
            difficulty=difficulty,
            topic=topic
        )
        
        try:
            # Get model parameters from config
            model_config = self.config.get('model', {})
            
            # Generate response with strict stop conditions
            response = model(
                prompt,
                max_tokens=model_config.get('max_tokens', 150),
                temperature=model_config.get('temperature', 0.4),
                top_p=model_config.get('top_p', 0.9),
                top_k=model_config.get('top_k', 40),
                repeat_penalty=model_config.get('repeat_penalty', 1.2),
                stop=["<|im_end|>", "<|im_start|>", "Student:", "User:"]
            )
            
            # Extract the generated text
            generated_text = response['choices'][0]['text'].strip()
            
            # Update conversation history
            self.conversation_history.append({
                "role": "user",
                "content": user_message
            })
            self.conversation_history.append({
                "role": "assistant", 
                "content": generated_text
            })
            
            # Keep history manageable (last 10 exchanges)
            if len(self.conversation_history) > 20:
                self.conversation_history = self.conversation_history[-20:]
            
            return generated_text
            
        except Exception as e:
            print(f"Error during inference: {e}")
            return self._generate_fallback_response(user_message)
    
    def clear_history(self):
        """Clear the conversation history."""
        self.conversation_history = []

    def get_history(self) -> List[Dict[str, str]]:
        """Return the current conversation history."""
        return self.conversation_history

    def generate_hint(self, context: Optional[str] = None) -> str:
        """Generate a hint based on the current conversation context."""
        model = model_loader.get_model()
        
        if model is None:
            return "💡 Try breaking down the problem into smaller parts. What do you already know for certain?"
        
        hint_prompt = self.prompt_builder.build_hint_prompt(
            conversation_history=self.conversation_history,
            context=context
        )
        
        try:
            model_config = self.config.get('model', {})
            
            response = model(
                hint_prompt,
                max_tokens=150,
                temperature=0.6,
                top_p=0.9,
                stop=["\n\n"]
            )
            
            return "💡 " + response['choices'][0]['text'].strip()
            
        except Exception as e:
            print(f"Error generating hint: {e}")
            return "💡 Consider what assumptions you might be making. Are they all necessary?"
    
    def _generate_fallback_response(self, user_message: str) -> str:
        """Generate a fallback Socratic response when the model is unavailable."""
        # More diverse and genuinely Socratic responses
        responses = [
            "That's a great starting point! Before I share more, what do you already know about this topic?",
            "Interesting question! Let me ask you this: what made you curious about this in the first place?",
            "I'd love to explore this with you. Can you break down what specifically you'd like to understand better?",
            "Good thinking! What's the simplest example of this concept you can think of?",
            "Let's approach this step by step. What's the first thing that comes to mind when you think about this?",
            "That's worth exploring! If you had to explain this to a friend, where would you start?",
            "Interesting! What patterns or connections do you notice in this problem?",
            "Let's dig into that. What would need to be true for your understanding to be correct?",
            "Great question! What's the most basic assumption we can start with?",
            "I see you're thinking deeply. What's the key concept you're trying to grasp here?",
        ]
        
        # Use a hash for more varied selection while still being deterministic
        index = (len(user_message) + sum(ord(c) for c in user_message[:20])) % len(responses)
        return responses[index]


# Global inference engine instance
inference_engine = InferenceEngine()
