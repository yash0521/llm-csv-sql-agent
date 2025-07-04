from pydantic import BaseModel, Field, validator
from typing import List, Dict, Any, Optional
from datetime import datetime

class QueryRequest(BaseModel):
    """Request model for CSV queries"""
    query: str = Field(..., description="Natural language query to execute on CSV data")
    file_id: str = Field(..., description="ID of the uploaded CSV file")
    limit: Optional[int] = Field(100, description="Maximum number of rows to return")
    
    @validator('query')
    def query_must_not_be_empty(cls, v):
        if not v.strip():
            raise ValueError('Query cannot be empty')
        return v.strip()
    
    @validator('limit')
    def limit_must_be_positive(cls, v):
        if v and v <= 0:
            raise ValueError('Limit must be positive')
        return v

class QueryResponse(BaseModel):
    """Response model for CSV query results"""
    query: str
    sql_generated: str
    result: List[Dict[str, Any]]
    execution_time: float
    row_count: int
    metadata: Optional[Dict[str, Any]] = None

class ChatRequest(BaseModel):
    """Request model for RAG chat"""
    message: str = Field(..., description="User message for RAG chat")
    conversation_id: Optional[str] = Field(None, description="Conversation ID for context")
    max_tokens: Optional[int] = Field(512, description="Maximum tokens in response")
    temperature: Optional[float] = Field(0.7, description="Response creativity (0-1)")
    
    @validator('message')
    def message_must_not_be_empty(cls, v):
        if not v.strip():
            raise ValueError('Message cannot be empty')
        return v.strip()
    
    @validator('temperature')
    def temperature_must_be_valid(cls, v):
        if v is not None and (v < 0 or v > 1):
            raise ValueError('Temperature must be between 0 and 1')
        return v

class ChatResponse(BaseModel):
    """Response model for RAG chat"""
    response: str
    model_used: str
    sources: List[Dict[str, Any]]
    cost_estimation: float
    response_time: float
    conversation_id: Optional[str] = None

class StreamChunk(BaseModel):
    """Model for streaming response chunks"""
    chunk_type: str = Field(..., description="Type of chunk: status, data, error, complete")
    content: Any = Field(..., description="Chunk content")
    timestamp: datetime = Field(default_factory=datetime.now)
    step: Optional[int] = None

class FileUploadResponse(BaseModel):
    """Response model for file uploads"""
    message: str
    file_id: str
    rows: int
    columns: List[str]
    sample_data: List[Dict[str, Any]]
    upload_time: datetime = Field(default_factory=datetime.now)

class HealthCheckResponse(BaseModel):
    """Response model for health checks"""
    status: str
    agents_loaded: Dict[str, bool]
    uploaded_files: int
    timestamp: datetime = Field(default_factory=datetime.now)

class FileListResponse(BaseModel):
    """Response model for file listing"""
    files: List[Dict[str, Any]]

class ErrorResponse(BaseModel):
    """Standard error response model"""
    error: str
    detail: Optional[str] = None
    error_code: Optional[str] = None
    timestamp: datetime = Field(default_factory=datetime.now)

# Additional models for internal use

class SQLGenerationResult(BaseModel):
    """Internal model for SQL generation results"""
    sql_query: str
    confidence: float
    explanation: str
    estimated_rows: Optional[int] = None

class RAGContext(BaseModel):
    """Internal model for RAG context"""
    query: str
    retrieved_documents: List[Dict[str, Any]]
    context_window: str
    relevance_scores: List[float]

class ModelRouting(BaseModel):
    """Internal model for LLM routing decisions"""
    query: str
    selected_model: str
    routing_reason: str
    estimated_cost: float
    complexity_score: float

class ConversationContext(BaseModel):
    """Model for maintaining conversation context"""
    conversation_id: str
    messages: List[Dict[str, str]]
    created_at: datetime = Field(default_factory=datetime.now)
    last_updated: datetime = Field(default_factory=datetime.now)
    
    def add_message(self, role: str, content: str):
        """Add a message to the conversation"""
        self.messages.append({
            "role": role,
            "content": content,
            "timestamp": datetime.now().isoformat()
        })
        self.last_updated = datetime.now()

# Configuration models

class LLMConfig(BaseModel):
    """Configuration for LLM services"""
    openai_api_key: str
    default_model: str = "gpt-3.5-turbo"
    max_tokens: int = 1000
    temperature: float = 0.7
    timeout: int = 30

class VectorStoreConfig(BaseModel):
    """Configuration for vector store"""
    collection_name: str = "rag_documents"
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    similarity_threshold: float = 0.7
    max_results: int = 5

class RouteLLMConfig(BaseModel):
    """Configuration for RouteLLM"""
    cheap_model: str = "gpt-3.5-turbo"
    expensive_model: str = "gpt-4"
    complexity_threshold: float = 0.5
    cost_optimization: bool = True
    performance_mode: bool = False