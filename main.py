from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import google.generativeai as genai
import os
import PyPDF2
from pathlib import Path
import chromadb
from chromadb.config import Settings
import hashlib
import asyncio
from typing import List, Optional
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
    allow_origins=["https://lcbf.vercel.app"],  # React dev server
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
PDF_FOLDER = "pdfs"

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
            return "नमस्ते! मैं LCB चैटबॉट हूं। मैं आपकी LCBFFertilizers से संबंधित प्रश्नों में सहायता कर सकता हूं।\n\nHello! I'm NavyaKosh ChatBot. I can help you with questions related to your LCB Fertilizers."
        
        elif any(word in prompt_lower for word in ["thank", "thanks", "धन्यवाद"]):
            return "आपका स्वागत है! कोई और प्रश्न है तो बेझिझक पूछें।\n\nYou're welcome! Feel free to ask if you have any other questions."
        
        else:
            return "मुझे खुशी होगी आपकी सहायता करने में। कृपया अपना प्रश्न PDF की जानकारी के संदर्भ में पूछें।\n\nI'd be happy to help you. Please ask your question related to the LCB Fertilizers Products and information."

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

# Initialize ChromaDB
try:
    client = chromadb.PersistentClient(path="./chroma_db")
    collection = client.get_or_create_collection(
        name="pdf_documents",
        metadata={"hnsw:space": "cosine"}
    )
    print("ChromaDB initialized successfully!")
except Exception as e:
    print(f"ChromaDB initialization error: {e}")
    client = chromadb.Client()
    collection = client.get_or_create_collection(name="pdf_documents")

class PDFChatBot:
    def __init__(self):
        self.pdf_folder = Path(PDF_FOLDER)
        self.pdf_folder.mkdir(exist_ok=True)
        self.conn = None
        
    async def setup_database(self):
        """Setup Neon database connection and tables"""
        try:
            if NEON_DATABASE_URL:
                self.conn = await asyncpg.connect(NEON_DATABASE_URL)
                
                await self.conn.execute('''
                    CREATE TABLE IF NOT EXISTS chat_history (
                        id SERIAL PRIMARY KEY,
                        user_message TEXT NOT NULL,
                        bot_response TEXT NOT NULL,
                        language VARCHAR(10) DEFAULT 'auto',
                        timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        sources TEXT[]
                    );
                ''')
                
                await self.conn.execute('''
                    CREATE TABLE IF NOT EXISTS pdf_documents (
                        id SERIAL PRIMARY KEY,
                        filename VARCHAR(255) NOT NULL,
                        file_hash VARCHAR(64) UNIQUE NOT NULL,
                        content_preview TEXT,
                        page_count INTEGER,
                        processed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    );
                ''')
                print("Database setup completed!")
            else:
                print("No database URL provided, skipping database setup")
                
        except Exception as e:
            print(f"Database setup error: {e}")
            self.conn = None
    
    def extract_text_from_pdf(self, pdf_path: Path) -> dict:
        """Extract text from PDF file"""
        try:
            text_content = ""
            with open(pdf_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                page_count = len(pdf_reader.pages)
                
                for page_num, page in enumerate(pdf_reader.pages):
                    page_text = page.extract_text()
                    if page_text.strip():
                        text_content += f"\n[Page {page_num + 1}]\n{page_text}\n"
            
            return {
                "content": text_content,
                "page_count": page_count,
                "filename": pdf_path.name
            }
        except Exception as e:
            print(f"Error extracting text from {pdf_path}: {e}")
            return None
    
    def chunk_text(self, text: str, chunk_size: int = 1000, overlap: int = 200) -> List[str]:
        """Split text into overlapping chunks"""
        words = text.split()
        chunks = []
        
        for i in range(0, len(words), chunk_size - overlap):
            chunk = ' '.join(words[i:i + chunk_size])
            if chunk.strip():
                chunks.append(chunk)
        
        return chunks
    
    async def process_pdfs(self):
        """Process all PDFs in the folder and store in vector database"""
        pdf_files = list(self.pdf_folder.glob("*.pdf"))
        
        if not pdf_files:
            print("No PDF files found in the pdfs folder")
            return
            
        for pdf_path in pdf_files:
            try:
                with open(pdf_path, 'rb') as f:
                    file_hash = hashlib.sha256(f.read()).hexdigest()
                
                try:
                    existing_docs = collection.get(where={"filename": pdf_path.name})
                    if existing_docs and len(existing_docs.get('ids', [])) > 0:
                        print(f"PDF {pdf_path.name} already processed, skipping...")
                        continue
                except Exception as check_error:
                    print(f"Error checking existing docs for {pdf_path.name}: {check_error}")
                
                print(f"Processing {pdf_path.name}...")
                
                pdf_data = self.extract_text_from_pdf(pdf_path)
                if not pdf_data or not pdf_data["content"].strip():
                    print(f"No text content found in {pdf_path.name}")
                    continue
                
                chunks = self.chunk_text(pdf_data["content"])
                
                if not chunks:
                    print(f"No chunks generated for {pdf_path.name}")
                    continue
                
                ids = []
                metadatas = []
                documents = []
                
                for i, chunk in enumerate(chunks):
                    chunk_id = f"{pdf_path.stem}_{i}"
                    ids.append(chunk_id)
                    documents.append(chunk)
                    metadatas.append({
                        "filename": pdf_path.name,
                        "chunk_index": i,
                        "file_hash": file_hash
                    })
                
                try:
                    collection.add(
                        documents=documents,
                        metadatas=metadatas,
                        ids=ids
                    )
                    print(f"Added {len(chunks)} chunks to ChromaDB for {pdf_path.name}")
                except Exception as add_error:
                    print(f"Error adding to ChromaDB: {add_error}")
                    continue
                
                if self.conn:
                    try:
                        await self.conn.execute('''
                            INSERT INTO pdf_documents (filename, file_hash, content_preview, page_count)
                            VALUES ($1, $2, $3, $4)
                            ON CONFLICT (file_hash) DO NOTHING
                        ''', pdf_path.name, file_hash, pdf_data["content"][:500], pdf_data["page_count"])
                    except Exception as db_error:
                        print(f"Error storing in database: {db_error}")
                
                print(f"Successfully processed {pdf_path.name}")
                
            except Exception as e:
                print(f"Error processing {pdf_path.name}: {e}")
    
    def search_relevant_content(self, query: str, n_results: int = 5) -> dict:
        """Search for relevant content in the vector database"""
        try:
            results = collection.query(
                query_texts=[query],
                n_results=n_results
            )
            
            print(f"Search results found: {len(results.get('documents', [[]])[0])} documents")
            
            if results.get('documents') and len(results['documents'][0]) > 0:
                return {
                    "documents": results['documents'][0],
                    "metadatas": results['metadatas'][0],
                    "distances": results.get('distances', [[]])[0]
                }
            
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
                greeting_response = "नमस्ते! मैं NavyaKosh हूं, आपका LCB Queries सहायक। मैं आपकी सेवा में हूं।"
            elif language == "english":
                language_instruction = "Please provide your complete answer in English with politeness and respect."
                greeting_response = "Hello! I'm NavyaKosh, your LCB windows Query assistant. I'm here to help you."
            else:
                language_instruction = "Please respond in the same language as the user's question with politeness and respect. If unclear, respond in both Hindi and English."
                greeting_response = "नमस्ते! Hello! I'm NavyaKosh, your LCB Query assistant. मैं आपका LCB Queries सहायक हूं।"
            
            # Handle greetings
            if is_general:
                general_prompt = f"""
You are NavyaKosh, a respectful and helpful PDF chatbot assistant.

User Query: {query}

Instructions:
- {language_instruction}
- Be very polite, respectful, and humble
- Introduce yourself as NavyaKosh if asked
- Show that you're here to help with LCB Fetilizers(Navyakosh)-related questions

Respond warmly and respectfully:
"""
                response_text, provider_used = self.generate_with_fallback(general_prompt)
            else:
                # Handle PDF-related queries
                if context:
                    prompt = f"""
You are NavyaKosh, a respectful LCB Fertilizers chatbot assistant.

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
You are LCB Fertilizer's (NavyaKosh), a respectful chatbot assistant.

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
        await chatbot.process_pdfs()
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
