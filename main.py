
import uvicorn
from fastapi import FastAPI
import google.generativeai as genai
import os
from dotenv import load_dotenv
from pydantic import BaseModel
from langchain_core.prompts import PromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.output_parsers import StrOutputParser
from langchain_community.document_loaders import WebBaseLoader
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS
from typing import List, Dict
from langchain_core.messages import HumanMessage, AIMessage
from starlette.middleware.cors import CORSMiddleware

# adding the CORS



# Load environment variables
load_dotenv()
api_key = os.getenv("GOOGLE_API_KEY")
genai.configure(api_key=api_key)

# Initialize Gemini model
model = genai.GenerativeModel("gemini-2.0-flash")

# Define the Web3 tutor prompt
web3_tutor_prompt = """
You are a friendly Web3 Tutor Bot. Your goals:
1. Explain blockchain concepts simply (e.g., "What's a smart contract?").
2. Use analogies (e.g., "Blockchain is like a digital ledger").
3. Give code examples where relevant (Solidity, Ethers.js).
4. Quiz users occasionally to reinforce learning.

Rules:
- If the user asks off-topic questions, steer back to Web3.
- Admit when you don't know something.
- Use emojis sparingly (e.g., 💡 for insights).

Context: {context}
Question: {input}
"""

# Initialize the chat



# Create prompt template
PROMPT = PromptTemplate(
    input_variables=["context", "input"],
    template=web3_tutor_prompt
)

# Initialize LangChain components
llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash", temperature=0.7)
output_parser = StrOutputParser()

# Create the chain
chain = PROMPT | llm | output_parser

# Initialize vector store for documentation
loader = WebBaseLoader("https://ethereum.org/en/developers/docs/")
data = loader.load()
embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
docsearch = FAISS.from_documents(data, embeddings)

def get_relevant_info(query: str) -> str:
    """Get relevant information from the vector store."""
    docs = docsearch.similarity_search(query, k=3)
    return "\n".join([doc.page_content for doc in docs])

class ChatRequest(BaseModel):
    """Request model for chat endpoints."""
    query: str

class ChatResponse(BaseModel):
    """Response model for chat endpoints."""
    response: str

# Initialize FastAPI app
app = FastAPI(title="Web3 Tutor Bot")

app.add_middleware(CORSMiddleware, allow_origins=["http://localhost:3000","http://localhost:5175", "http://co-insyde.onrender.com"],  allow_credentials=True, allow_methods=["*"], allow_headers=["*"])

@app.post("/chat", response_model=ChatResponse)
async def chat_endpoint(request: ChatRequest):
    """Endpoint for simple chat without memory."""
    try:
        context = get_relevant_info(request.query)
        response = chain.invoke({
            "context": context,
            "input": request.query
        })
        return ChatResponse(response=response)
    except Exception as e:
        return ChatResponse(response=f"Error processing request: {str(e)}")

@app.post("/chat_with_context", response_model=ChatResponse)
async def chat_with_context(request: ChatRequest):
    """Endpoint for chat with context retrieval."""
    try:
        chat = model.start_chat(history=[])
        context = get_relevant_info(request.query)
        response = chat.send_message(f"Context:\n{context}\n\nQuestion: {request.query}")
        return ChatResponse(response=response.text)
    except Exception as e:
        return ChatResponse(response=f"Error processing request: {str(e)}")
#chat history for previous questions
chat_history = []



if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8008)