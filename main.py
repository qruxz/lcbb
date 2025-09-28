from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import google.generativeai as genai
import os

import asyncio
from typing import List, Optional
import asyncpg
import json
from datetime import datetime
from dotenv import load_dotenv
import requests
import time

# Initialize FastAPI app
app = FastAPI(title="NavyaKosh PDF Chatbot API")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],  # React dev server
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Pydantic models
class ChatMessage(BaseModel):
    message: str
    language: str = "auto"

class ChatResponse(BaseModel):
    response: str
    sources: List[str] = []

load_dotenv()
# Configuration
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
NEON_DATABASE_URL = os.getenv("NEON_DATABASE_URL")

# LLM Provider Classes
class GeminiLLM:
    def __init__(self, api_key: str):
        self.api_key = api_key
        genai.configure(api_key=api_key)
        
        # Try different model names in order
        self.models_to_try = [
            'gemini-1.5-flash-latest',
            'gemini-1.5-flash-001', 
            'gemini-1.5-flash',
            'gemini-pro-latest',
            'gemini-pro'
        ]
        
        self.model = None
        self._initialize_model()
    
    def _initialize_model(self):
        """Initialize the first working Gemini model"""
        for model_name in self.models_to_try:
            try:
                test_model = genai.GenerativeModel(model_name)
                # Quick test
                test_response = test_model.generate_content("Hello")
                if test_response and test_response.text:
                    self.model = test_model
                    print(f"✅ Using Gemini model: {model_name}")
                    return
            except Exception as e:
                print(f"❌ Failed to initialize {model_name}: {e}")
                continue
        
        # If all fail, raise exception
        raise Exception("No working Gemini model found")
    
    def generate(self, prompt: str) -> str:
        if not self.model:
            return None
            
        try:
            response = self.model.generate_content(prompt)
            return response.text if response and response.text else None
        except Exception as e:
            print(f"❌ Gemini generation failed: {e}")
            return None

class OpenAILLM:
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.base_url = "https://api.openai.com/v1/chat/completions"
        print("🧠 OpenAI backup ready")
    
    def generate(self, prompt: str) -> str:
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        data = {
            "model": "gpt-3.5-turbo",
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": 500,
            "temperature": 0.7
        }
        
        try:
            response = requests.post(self.base_url, headers=headers, json=data, timeout=30)
            if response.status_code == 200:
                return response.json()["choices"][0]["message"]["content"]
            else:
                print(f"❌ OpenAI API error: {response.status_code}")
                return None
        except Exception as e:
            print(f"❌ OpenAI generation failed: {e}")
            return None

class GroqLLM:
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.base_url = "https://api.groq.com/openai/v1/chat/completions"
        print("⚡ Groq backup ready")
    
    def generate(self, prompt: str) -> str:
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        data = {
            "model": "llama3-8b-8192",
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": 500,
            "temperature": 0.7
        }
        
        try:
            response = requests.post(self.base_url, headers=headers, json=data, timeout=30)
            if response.status_code == 200:
                return response.json()["choices"][0]["message"]["content"]
            else:
                print(f"❌ Groq API error: {response.status_code}")
                return None
        except Exception as e:
            print(f"❌ Groq generation failed: {e}")
            return None

class FallbackLLM:
    def __init__(self):
        print("🛟 Template fallback ready")
    
    def generate(self, prompt: str) -> str:
        # Extract query from prompt
        prompt_lower = prompt.lower()
        
        # Simple template-based responses
        if any(word in prompt_lower for word in ["hello", "hi", "namaste", "नमस्ते"]):
            return "नमस्ते! मैं NavyaKosh चैटबॉट हूं। मैं आपकी PDF से संबंधित प्रश्नों में सहायता कर सकता हूं।\n\nHello! I'm NavyaKosh ChatBot. I can help you with questions related to your PDFs."
        
        elif any(word in prompt_lower for word in ["thank", "thanks", "धन्यवाद"]):
            return "आपका स्वागत है! कोई और प्रश्न है तो बेझिझक पूछें।\n\nYou're welcome! Feel free to ask if you have any other questions."
        
        else:
            return "मुझे खुशी होगी आपकी सहायता करने में। कृपया अपना प्रश्न PDF की जानकारी के संदर्भ में पूछें।\n\nI'd be happy to help you. Please ask your question related to the PDF information."

# Initialize LLM providers
def initialize_llm_providers():
    """Initialize all available LLM providers"""
    providers = []
    
    # Try Gemini first (primary)
    if GEMINI_API_KEY:
        try:
            gemini_llm = GeminiLLM(GEMINI_API_KEY)
            providers.append(("gemini", gemini_llm))
        except Exception as e:
            print(f"❌ Gemini initialization failed: {e}")
    
    # Add OpenAI backup
    if OPENAI_API_KEY:
        try:
            openai_llm = OpenAILLM(OPENAI_API_KEY)
            providers.append(("openai", openai_llm))
        except Exception as e:
            print(f"❌ OpenAI initialization failed: {e}")
    
    # Add Groq backup
    if GROQ_API_KEY:
        try:
            groq_llm = GroqLLM(GROQ_API_KEY)
            providers.append(("groq", groq_llm))
        except Exception as e:
            print(f"❌ Groq initialization failed: {e}")
    
    # Always add fallback
    providers.append(("fallback", FallbackLLM()))
    
    print(f"🚀 Initialized {len(providers)} LLM providers")
    return providers

# Initialize all providers
llm_providers = initialize_llm_providers()

class PDFChatBot:
    def __init__(self):
        self.conn = None
        
    async def setup_database(self):
        """Setup Neon database connection"""
        try:
            if NEON_DATABASE_URL:
                self.conn = await asyncpg.connect(NEON_DATABASE_URL)
                print("Database connection established!")
            else:
                print("No database URL provided, skipping database connection")
                
        except Exception as e:
            print(f"Database setup error: {e}")
            self.conn = None
    
    def search_relevant_content(self, query: str, n_results: int = 5) -> dict:
        """Search for relevant content in Neon database"""
        try:
            if not self.conn:
                return {"documents": [], "metadatas": [], "distances": []}
            
            # Simple text search in Neon database - you can enhance this with vector search later
            # For now, return empty results to avoid ChromaDB dependency
            print(f"Search query: {query}")
            return {"documents": [], "metadatas": [], "distances": []}
            
        except Exception as e:
            print(f"Search error: {e}")
            return {"documents": [], "metadatas": [], "distances": []}
    
    def is_greeting_or_general(self, query: str) -> bool:
        """Check if the query is a greeting or general conversation"""
        query_lower = query.lower().strip()
        greetings = [
            'hello', 'hi', 'hey', 'namaste', 'namaskar', 'good morning', 'good evening',
            'how are you', 'what is your name', 'who are you', 'about you', 'thank you',
            'thanks', 'dhanyawad', 'shukriya', 'bye', 'goodbye', 'alvida'
        ]
        
        for greeting in greetings:
            if greeting in query_lower:
                return True
        return False
    
    def generate_with_fallback(self, prompt: str) -> tuple[str, str]:
        """Generate response with fallback system"""
        for provider_name, provider in llm_providers:
            try:
                print(f"🔄 Trying {provider_name}...")
                response = provider.generate(prompt)
                
                if response and len(response.strip()) > 10:
                    print(f"✅ Success with {provider_name}")
                    return response, provider_name
                else:
                    print(f"⚠️ {provider_name} returned empty/short response")
                    
            except Exception as e:
                print(f"❌ {provider_name} failed: {e}")
                continue
        
        # Ultimate fallback
        print("🛟 Using ultimate fallback")
        return "मुझे खेद है, मुझे तकनीकी समस्या हो रही है। कृपया बाद में पुनः प्रयास करें।\n\nI'm sorry, I'm experiencing technical difficulties. Please try again later.", "ultimate_fallback"
    
    async def generate_response(self, query: str, language: str = "auto") -> dict:
        """Generate response using multiple LLM providers with fallback"""
        try:
            is_general = self.is_greeting_or_general(query)
            search_results = self.search_relevant_content(query)
            
            context = ""
            sources = []
            
            if search_results["documents"] and not is_general:
                context = "\n\n".join(search_results["documents"][:3])
                sources = list(set([meta["filename"] for meta in search_results["metadatas"][:3]]))
            
            # Prepare language-specific instructions
            if language == "hindi":
                language_instruction = "कृपया अपना पूरा उत्तर हिंदी में विनम्रता और सम्मान के साथ दें।"
                greeting_response = "नमस्ते! मैं NavyaKosh हूं, आपका PDF सहायक। मैं आपकी सेवा में हूं।"
            elif language == "english":
                language_instruction = "Please provide your complete answer in English with politeness and respect."
                greeting_response = "Hello! I'm NavyaKosh, your PDF assistant. I'm here to help you."
            else:
                language_instruction = "Please respond in the same language as the user's question with politeness and respect. If unclear, respond in both Hindi and English."
                greeting_response = "नमस्ते! Hello! I'm NavyaKosh, your PDF assistant. मैं आपका PDF सहायक हूं।"
            
            # Handle greetings
            if is_general:
                general_prompt = f"""
You are NavyaKosh, a respectful and helpful PDF chatbot assistant.

User Query: {query}

Instructions:
- {language_instruction}
- Be very polite, respectful, and humble
- Introduce yourself as NavyaKosh if asked
- Show that you're here to help with PDF-related questions

Respond warmly and respectfully:
"""
                response_text, provider_used = self.generate_with_fallback(general_prompt)
            else:
                # Handle PDF-related queries
                if context:
                    prompt = f"""
You are NavyaKosh, a respectful PDF chatbot assistant.

Relevant PDF information:
{context}

User Question: {query}

Instructions:
- {language_instruction}
- Be respectful, polite, and helpful
- Use the provided PDF context to give accurate answers
- Start with respectful greetings
- Thank the user and offer further assistance

Provide a detailed, respectful response:
"""
                else:
                    prompt = f"""
You are NavyaKosh, a respectful PDF chatbot assistant.

User Question: {query}

Instructions:
- {language_instruction}
- Be respectful and polite
- Explain that you don't have specific information about their question
- Apologize politely and offer to help with other questions

Respond politely:
"""
                
                response_text, provider_used = self.generate_with_fallback(prompt)
            
            # Store chat history
            if self.conn:
                try:
                    await self.conn.execute('''
                        INSERT INTO chat_history (user_message, bot_response, language, sources)
                        VALUES ($1, $2, $3, $4)
                    ''', query, response_text, language, sources)
                except Exception as db_error:
                    print(f"Error storing chat history: {db_error}")
            
            return {
                "response": response_text,
                "sources": sources,
                "provider_used": provider_used
            }
            
        except Exception as e:
            print(f"Response generation error: {e}")
            error_msg = "मुझे खेद है, मुझे समस्या हो रही है। कृपया पुनः प्रयास करें।\n\nI apologize, I'm having trouble. Please try again."
            return {
                "response": error_msg,
                "sources": []
            }

# Initialize chatbot
chatbot = PDFChatBot()

@app.on_event("startup")
async def startup_event():
    """Initialize the application"""
    try:
        await chatbot.setup_database()
        print("Application initialized successfully!")
    except Exception as e:
        print(f"Startup error: {e}")

@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown"""
    if hasattr(chatbot, 'conn') and chatbot.conn:
        await chatbot.conn.close()

@app.get("/")
async def root():
    return {"message": "NavyaKosh PDF Chatbot API is running with multiple LLM fallbacks!"}

@app.post("/chat", response_model=ChatResponse)
async def chat(message: ChatMessage):
    """Main chat endpoint with fallback support"""
    try:
        result = await chatbot.generate_response(message.message, message.language)
        return ChatResponse(response=result["response"], sources=result["sources"])
    except Exception as e:
        print(f"Chat endpoint error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    return {
        "status": "healthy", 
        "timestamp": datetime.now().isoformat(),
        "providers": len(llm_providers)
    }

@app.get("/pdfs")
async def list_pdfs():
    """Get list of processed PDFs"""
    try:
        if chatbot.conn:
            pdfs = await chatbot.conn.fetch("SELECT filename, page_count, processed_at FROM pdf_documents ORDER BY processed_at DESC")
            return [dict(pdf) for pdf in pdfs]
        else:
            try:
                all_docs = collection.get()
                unique_files = set()
                if all_docs.get('metadatas'):
                    for meta in all_docs['metadatas']:
                        unique_files.add(meta.get('filename', 'Unknown'))
                return [{"filename": f, "page_count": "N/A", "processed_at": "N/A"} for f in unique_files]
            except:
                return []
    except Exception as e:
        print(f"Error listing PDFs: {e}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    print(f"🚀 Starting with {len(llm_providers)} LLM providers")
    uvicorn.run(app, host="0.0.0.0", port=8000)
