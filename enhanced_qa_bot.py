"""
Enhanced Q&A Bot with Conversation History
----------------------------------------
This module provides an improved Q&A bot with better conversation history management.
"""

from typing import Dict, Any, List, Optional
from datetime import datetime
import json
import os
from pathlib import Path
from dataclasses import dataclass, asdict

from langchain_core.messages import HumanMessage, AIMessage, BaseMessage
from langchain.memory import ChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.chains import LLMChain
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
os.environ["GOOGLE_API_KEY"] = os.getenv('GOOGLE_API_KEY')

@dataclass
class Message:
    """Represents a single message in the conversation."""
    content: str
    sender: str  # 'user' or 'ai'
    timestamp: str
    metadata: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert message to dictionary."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Message':
        """Create message from dictionary."""
        return cls(**data)

class ChatHistoryStore:
    """Manages chat histories with persistence to disk."""
    
    def __init__(self, storage_path: str = "./chat_histories"):
        """Initialize with storage path."""
        self.storage_path = Path(storage_path)
        self.storage_path.mkdir(parents=True, exist_ok=True)
        self.memories: Dict[str, ChatMessageHistory] = {}
    
    def _get_storage_file(self, session_id: str) -> Path:
        """Get the storage file path for a session."""
        return self.storage_path / f"{session_id}.json"
    
    def get_chat_history(self, session_id: str) -> ChatMessageHistory:
        """Get or create chat history for a session."""
        if session_id not in self.memories:
            # Try to load from disk
            storage_file = self._get_storage_file(session_id)
            if storage_file.exists():
                with open(storage_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                history = ChatMessageHistory()
                for msg_data in data:
                    if msg_data['type'] == 'human':
                        history.add_user_message(msg_data['content'])
                    else:
                        history.add_ai_message(msg_data['content'])
                self.memories[session_id] = history
            else:
                self.memories[session_id] = ChatMessageHistory()
        return self.memories[session_id]
    
    def save_chat_history(self, session_id: str) -> None:
        """Save chat history to disk."""
        if session_id in self.memories:
            storage_file = self._get_storage_file(session_id)
            messages = []
            for msg in self.memories[session_id].messages:
                msg_type = 'human' if isinstance(msg, HumanMessage) else 'ai'
                messages.append({
                    'type': msg_type,
                    'content': msg.content,
                    'timestamp': datetime.now().isoformat()
                })
            with open(storage_file, 'w', encoding='utf-8') as f:
                json.dump(messages, f, indent=2)
    
    def clear_history(self, session_id: str) -> None:
        """Clear chat history for a session."""
        if session_id in self.memories:
            self.memories[session_id].clear()
        # Also delete from disk
        storage_file = self._get_storage_file(session_id)
        if storage_file.exists():
            storage_file.unlink()

class QABot:
    """Enhanced Q&A Bot with conversation history and persistence."""
    
    def __init__(self, model_name: str = "gemini-1.5-flash", temperature: float = 0.7):
        """Initialize the Q&A bot."""
        self.llm = ChatGoogleGenerativeAI(
            model=model_name,
            temperature=temperature,
            google_api_key=os.getenv('GOOGLE_API_KEY')
        )
        self.history_store = ChatHistoryStore()
        self._setup_chain()
    
    def _setup_chain(self) -> None:
        """Set up the LLM chain with history."""
        # System message with instructions
        system_prompt = """You are a helpful, knowledgeable, and friendly AI assistant. 
        Keep your responses concise and to the point. If you don't know something, 
        just say you don't know instead of making up an answer."""
        
        # Create the prompt with history
        prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            MessagesPlaceholder(variable_name="history"),
            ("human", "{input}")
        ])
        
        # Create the chain
        self.chain = LLMChain(llm=self.llm, prompt=prompt)
        
        # Add history
        self.chain_with_history = RunnableWithMessageHistory(
            self.chain,
            lambda session_id: self.history_store.get_chat_history(session_id),
            input_messages_key="input",
            history_messages_key="history"
        )
    
    def get_response(self, user_input: str, session_id: str = "default") -> str:
        """
        Get a response from the AI assistant.
        
        Args:
            user_input: The user's input/message
            session_id: Unique identifier for the conversation session
            
        Returns:
            str: The AI's response
        """
        try:
            # Get response with history
            response = self.chain_with_history.invoke(
                {"input": user_input},
                config={"configurable": {"session_id": session_id}}
            )
            
            # Save the conversation
            self.history_store.save_chat_history(session_id)
            
            return response["text"].strip()
            
        except Exception as e:
            return f"Sorry, I encountered an error: {str(e)}"
    
    def clear_conversation(self, session_id: str = "default") -> None:
        """Clear the conversation history for a session."""
        self.history_store.clear_history(session_id)
    
    def get_conversation_history(self, session_id: str = "default") -> List[Dict[str, Any]]:
        """Get the conversation history for a session."""
        history = self.history_store.get_chat_history(session_id)
        return [
            {
                'role': 'user' if isinstance(msg, HumanMessage) else 'ai',
                'content': msg.content,
                'timestamp': datetime.now().isoformat()
            }
            for msg in history.messages
        ]

# Example usage
if __name__ == "__main__":
    # Initialize the bot
    bot = QABot()
    
    # Example conversation
    session_id = "example_session_123"
    
    # First message
    response = bot.get_response("Hello! How are you today?", session_id)
    print(f"User: Hello! How are you today?")
    print(f"AI: {response}\n")
    
    # Second message with context
    response = bot.get_response("What was my previous message?", session_id)
    print(f"User: What was my previous message?")
    print(f"AI: {response}\n")
    
    # Get conversation history
    print("Conversation History:")
    for msg in bot.get_conversation_history(session_id):
        print(f"{msg['role'].upper()}: {msg['content']}")
    
    # Clear the conversation
    bot.clear_conversation(session_id)
    print("\nConversation history cleared.")
