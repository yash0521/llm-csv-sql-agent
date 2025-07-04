from fastapi import FastAPI, UploadFile, File, HTTPException, BackgroundTasks
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
import sqlite3
import json
import asyncio
from typing import AsyncGenerator, Optional, List, Dict
import io
import os
from contextlib import asynccontextmanager

from app.models.schemas import QueryRequest, ChatRequest, QueryResponse
from app.agents.csv_sql_agent import CSVSQLAgent
from app.agents.rag_agent import RAGAgent
from app.services.llm_service import LLMService
from app.utils.route_llm import RouteLLM

# Global variables for agents
csv_agent = None
rag_agent = None
route_llm = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    global csv_agent, rag_agent, route_llm
    
    # Initialize services
    llm_service = LLMService()
    csv_agent = CSVSQLAgent(llm_service)
    rag_agent = RAGAgent(llm_service)
    route_llm = RouteLLM()
    
    print("✅ All agents initialized successfully")
    yield
    
    # Shutdown
    print("🔄 Shutting down application...")

app = FastAPI(
    title="LLM CSV-to-SQL Agent & RAG Chat System",
    description="Advanced LLM-based system for CSV querying and RAG chat with routing optimization",
    version="1.0.0",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global storage for uploaded CSV files
uploaded_files: Dict[str, pd.DataFrame] = {}

@app.get("/")
async def root():
    return {
        "message": "LLM CSV-to-SQL Agent & RAG Chat System",
        "status": "running",
        "endpoints": {
            "upload_csv": "/upload-csv/",
            "query_csv": "/query-csv/",
            "stream_query": "/stream-query/",
            "rag_chat": "/rag-chat/",
            "stream_chat": "/stream-chat/"
        }
    }

@app.post("/upload-csv/")
async def upload_csv(file: UploadFile = File(...)):
    """Upload and process CSV file for querying"""
    try:
        if not file.filename.endswith('.csv'):
            raise HTTPException(status_code=400, detail="Only CSV files are supported")
        
        # Read CSV content
        content = await file.read()
        df = pd.read_csv(io.StringIO(content.decode('utf-8')))
        
        # Store in memory (in production, use proper storage)
        file_id = file.filename.replace('.csv', '')
        uploaded_files[file_id] = df
        
        # Create SQLite table for SQL queries
        conn = sqlite3.connect(':memory:')
        df.to_sql(file_id, conn, index=False, if_exists='replace')
        
        return {
            "message": f"CSV file '{file.filename}' uploaded successfully",
            "file_id": file_id,
            "rows": len(df),
            "columns": list(df.columns),
            "sample_data": df.head(3).to_dict('records')
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing CSV: {str(e)}")

@app.post("/query-csv/")
async def query_csv(request: QueryRequest):
    """Execute natural language query on uploaded CSV"""
    try:
        if request.file_id not in uploaded_files:
            raise HTTPException(status_code=404, detail="File not found. Please upload CSV first.")
        
        df = uploaded_files[request.file_id]
        result = await csv_agent.process_query(request.query, df, request.file_id)
        
        return QueryResponse(
            query=request.query,
            sql_generated=result.get('sql', ''),
            result=result.get('data', []),
            execution_time=result.get('execution_time', 0),
            row_count=len(result.get('data', []))
        )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Query execution failed: {str(e)}")

@app.post("/stream-query/")
async def stream_query(request: QueryRequest):
    """Stream SQL query results in real-time"""
    
    async def generate_response() -> AsyncGenerator[str, None]:
        try:
            if request.file_id not in uploaded_files:
                yield f"data: {json.dumps({'error': 'File not found'})}\n\n"
                return
            
            df = uploaded_files[request.file_id]
            
            # Stream the SQL generation and execution process
            yield f"data: {json.dumps({'status': 'Analyzing query...', 'step': 1})}\n\n"
            await asyncio.sleep(0.1)
            
            yield f"data: {json.dumps({'status': 'Generating SQL...', 'step': 2})}\n\n"
            await asyncio.sleep(0.1)
            
            # Process query with streaming
            async for chunk in csv_agent.process_query_stream(request.query, df, request.file_id):
                yield f"data: {json.dumps(chunk)}\n\n"
                await asyncio.sleep(0.05)  # Small delay for streaming effect
            
            yield f"data: {json.dumps({'status': 'complete'})}\n\n"
            
        except Exception as e:
            yield f"data: {json.dumps({'error': str(e)})}\n\n"
    
    return StreamingResponse(
        generate_response(),
        media_type="text/plain",
        headers={"Cache-Control": "no-cache", "Connection": "keep-alive"}
    )

@app.post("/rag-chat/")
async def rag_chat(request: ChatRequest):
    """RAG-based chat with RouteLLM optimization"""
    try:
        # Route to appropriate model based on query complexity
        selected_model = route_llm.route_query(request.message)
        
        # Process with RAG
        response = await rag_agent.chat(
            message=request.message,
            conversation_id=request.conversation_id,
            model=selected_model
        )
        
        return {
            "response": response["answer"],
            "model_used": selected_model,
            "sources": response.get("sources", []),
            "cost_estimation": route_llm.estimate_cost(request.message, selected_model),
            "response_time": response.get("response_time", 0)
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Chat processing failed: {str(e)}")

@app.post("/stream-chat/")
async def stream_chat(request: ChatRequest):
    """Stream RAG chat responses in real-time"""
    
    async def generate_chat_response() -> AsyncGenerator[str, None]:
        try:
            # Route query
            selected_model = route_llm.route_query(request.message)
            
            yield f"data: {json.dumps({'status': 'routing', 'model': selected_model})}\n\n"
            await asyncio.sleep(0.1)
            
            # Stream RAG response
            async for chunk in rag_agent.chat_stream(
                message=request.message,
                conversation_id=request.conversation_id,
                model=selected_model
            ):
                yield f"data: {json.dumps(chunk)}\n\n"
                await asyncio.sleep(0.05)
            
            yield f"data: {json.dumps({'status': 'complete'})}\n\n"
            
        except Exception as e:
            yield f"data: {json.dumps({'error': str(e)})}\n\n"
    
    return StreamingResponse(
        generate_chat_response(),
        media_type="text/plain",
        headers={"Cache-Control": "no-cache", "Connection": "keep-alive"}
    )

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "agents_loaded": {
            "csv_agent": csv_agent is not None,
            "rag_agent": rag_agent is not None,
            "route_llm": route_llm is not None
        },
        "uploaded_files": len(uploaded_files)
    }

@app.get("/files")
async def list_files():
    """List all uploaded CSV files"""
    return {
        "files": [
            {
                "file_id": file_id,
                "rows": len(df),
                "columns": list(df.columns)
            }
            for file_id, df in uploaded_files.items()
        ]
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )