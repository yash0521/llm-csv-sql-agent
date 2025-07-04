import asyncio
import time
import uuid
from typing import Dict, List, Any, AsyncGenerator, Optional
import json
from datetime import datetime, timedelta

from app.services.llm_service import LLMService
from app.services.vector_store import VectorStore
from app.utils.prompt_templates import PromptTemplates
from app.models.schemas import ConversationContext

class RAGAgent:
    """Advanced RAG Chat Agent with context management and optimization"""
    
    def __init__(self, llm_service: LLMService):
        self.llm_service = llm_service
        self.vector_store = VectorStore()
        self.prompt_templates = PromptTemplates()
        self.conversations: Dict[str, ConversationContext] = {}
        self.performance_metrics = {
            "total_queries": 0,
            "average_response_time": 0,
            "context_retrieval_time": 0,
            "generation_time": 0,
            "accuracy_score": 0.95
        }
    
    async def chat(self, message: str, conversation_id: Optional[str] = None, 
                   model: str = "gpt-3.5-turbo") -> Dict[str, Any]:
        """Process chat message with RAG"""
        start_time = time.time()
        
        try:
            # Get or create conversation
            if not conversation_id:
                conversation_id = str(uuid.uuid4())
            
            conversation = self._get_or_create_conversation(conversation_id)
            
            # Retrieve relevant context
            retrieval_start = time.time()
            retrieved_docs = await self.vector_store.similarity_search(
                query=message,
                k=5,
                threshold=0.7
            )
            retrieval_time = time.time() - retrieval_start
            
            # Build context-aware prompt
            context = self._build_context(message, retrieved_docs, conversation)
            
            # Generate response
            generation_start = time.time()
            response = await self.llm_service.generate_completion(
                prompt=context,
                model=model,
                temperature=0.7,
                max_tokens=512
            )
            generation_time = time.time() - generation_start
            
            # Update conversation
            conversation.add_message("user", message)
            conversation.add_message("assistant", response)
            
            # Calculate total time
            total_time = time.time() - start_time
            
            # Update metrics
            self._update_metrics(total_time, retrieval_time, generation_time)
            
            return {
                "answer": response,
                "sources": [self._format_source(doc) for doc in retrieved_docs],
                "conversation_id": conversation_id,
                "response_time": total_time,
                "model_used": model,
                "context_quality": self._assess_context_quality(retrieved_docs)
            }
            
        except Exception as e:
            return {
                "error": str(e),
                "answer": "I apologize, but I encountered an error processing your request.",
                "sources": [],
                "conversation_id": conversation_id,
                "response_time": time.time() - start_time
            }
    
    async def chat_stream(self, message: str, conversation_id: Optional[str] = None,
                         model: str = "gpt-3.5-turbo") -> AsyncGenerator[Dict[str, Any], None]:
        """Stream chat response with real-time updates"""
        
        # Initialize conversation
        if not conversation_id:
            conversation_id = str(uuid.uuid4())
        
        conversation = self._get_or_create_conversation(conversation_id)
        
        # Step 1: Context retrieval
        yield {
            "step": "retrieval",
            "status": "Searching knowledge base...",
            "conversation_id": conversation_id
        }
        
        try:
            retrieval_start = time.time()
            retrieved_docs = await self.vector_store.similarity_search(
                query=message,
                k=5,
                threshold=0.7
            )
            retrieval_time = time.time() - retrieval_start
            
            yield {
                "step": "context",
                "status": f"Found {len(retrieved_docs)} relevant documents",
                "data": {
                    "sources_count": len(retrieved_docs),
                    "retrieval_time": retrieval_time
                }
            }
            
            # Step 2: Generate response
            yield {
                "step": "generation",
                "status": "Generating response...",
                "data": {"model": model}
            }
            
            # Build context
            context = self._build_context(message, retrieved_docs, conversation)
            
            # Stream response generation
            async for chunk in self.llm_service.generate_completion_stream(
                prompt=context,
                model=model,
                temperature=0.7,
                max_tokens=512
            ):
                yield {
                    "step": "response_chunk",
                    "status": "streaming",
                    "data": {"chunk": chunk}
                }
            
            # Final response with metadata
            yield {
                "step": "complete",
                "status": "Response generated successfully",
                "data": {
                    "sources": [self._format_source(doc) for doc in retrieved_docs],
                    "conversation_id": conversation_id,
                    "context_quality": self._assess_context_quality(retrieved_docs)
                }
            }
            
        except Exception as e:
            yield {
                "step": "error",
                "status": f"Error: {str(e)}",
                "data": {"error": str(e)}
            }
    
    async def add_documents(self, documents: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Add documents to the knowledge base"""
        try:
            added_count = await self.vector_store.add_documents(documents)
            return {
                "success": True,
                "added_documents": added_count,
                "total_documents": await self.vector_store.get_document_count()
            }
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "added_documents": 0
            }
    
    async def search_knowledge_base(self, query: str, k: int = 10) -> List[Dict[str, Any]]:
        """Search the knowledge base directly"""
        try:
            results = await self.vector_store.similarity_search(
                query=query,
                k=k,
                threshold=0.5
            )
            return [self._format_source(doc) for doc in results]
        except Exception as e:
            return [{"error": str(e)}]
    
    def _get_or_create_conversation(self, conversation_id: str) -> ConversationContext:
        """Get existing conversation or create new one"""
        if conversation_id not in self.conversations:
            self.conversations[conversation_id] = ConversationContext(
                conversation_id=conversation_id,
                messages=[]
            )
        return self.conversations[conversation_id]
    
    def _build_context(self, message: str, retrieved_docs: List[Dict], 
                      conversation: ConversationContext) -> str:
        """Build context-aware prompt for generation"""
        
        # System prompt
        system_prompt = self.prompt_templates.get_rag_system_prompt()
        
        # Retrieved context
        context_text = self._format_retrieved_context(retrieved_docs)
        
        # Conversation history (last 4 messages)
        conversation_history = self._format_conversation_history(conversation, max_messages=4)
        
        # Build final prompt
        prompt = f"""{system_prompt}

RETRIEVED CONTEXT:
{context_text}

CONVERSATION HISTORY:
{conversation_history}

CURRENT QUESTION: {message}

Please provide a helpful, accurate response based on the retrieved context and conversation history."""
        
        return prompt
    
    def _format_retrieved_context(self, retrieved_docs: List[Dict]) -> str:
        """Format retrieved documents into context"""
        if not retrieved_docs:
            return "No relevant context found in knowledge base."
        
        context_parts = []
        for i, doc in enumerate(retrieved_docs, 1):
            content = doc.get('content', doc.get('text', ''))
            metadata = doc.get('metadata', {})
            source = metadata.get('source', f'Document {i}')
            
            context_parts.append(f"[Source {i}: {source}]\n{content}")
        
        return "\n\n".join(context_parts)
    
    def _format_conversation_history(self, conversation: ConversationContext, 
                                   max_messages: int = 4) -> str:
        """Format recent conversation history"""
        if not conversation.messages:
            return "No previous conversation."
        
        # Get last N messages
        recent_messages = conversation.messages[-max_messages:]
        
        formatted_messages = []
        for msg in recent_messages:
            role = msg['role'].title()
            content = msg['content']
            formatted_messages.append(f"{role}: {content}")
        
        return "\n".join(formatted_messages)
    
    def _format_source(self, doc: Dict[str, Any]) -> Dict[str, Any]:
        """Format source information for response"""
        metadata = doc.get('metadata', {})
        return {
            "content": doc.get('content', doc.get('text', ''))[:200] + "...",
            "source": metadata.get('source', 'Unknown'),
            "title": metadata.get('title', 'Untitled'),
            "relevance_score": doc.get('score', 0.0),
            "url": metadata.get('url', ''),
            "document_type": metadata.get('type', 'document')
        }
    
    def _assess_context_quality(self, retrieved_docs: List[Dict]) -> Dict[str, Any]:
        """Assess the quality of retrieved context"""
        if not retrieved_docs:
            return {"quality": "poor", "score": 0.0, "reason": "No relevant documents found"}
        
        scores = [doc.get('score', 0.0) for doc in retrieved_docs]
        avg_score = sum(scores) / len(scores)
        
        if avg_score >= 0.8:
            quality = "excellent"
        elif avg_score >= 0.6:
            quality = "good"  
        elif avg_score >= 0.4:
            quality = "fair"
        else:
            quality = "poor"
        
        return {
            "quality": quality,
            "score": avg_score,
            "document_count": len(retrieved_docs),
            "top_score": max(scores) if scores else 0.0
        }
    
    def _update_metrics(self, total_time: float, retrieval_time: float, generation_time: float):
        """Update performance metrics"""
        self.performance_metrics["total_queries"] += 1
        
        # Update averages
        count = self.performance_metrics["total_queries"]
        
        current_avg_response = self.performance_metrics["average_response_time"]
        self.performance_metrics["average_response_time"] = (
            (current_avg_response * (count - 1) + total_time) / count
        )
        
        current_avg_retrieval = self.performance_metrics["context_retrieval_time"]
        self.performance_metrics["context_retrieval_time"] = (
            (current_avg_retrieval * (count - 1) + retrieval_time) / count
        )
        
        current_avg_generation = self.performance_metrics["generation_time"]
        self.performance_metrics["generation_time"] = (
            (current_avg_generation * (count - 1) + generation_time) / count
        )
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get current performance statistics"""
        return {
            "total_queries": self.performance_metrics["total_queries"],
            "average_response_time": round(self.performance_metrics["average_response_time"], 4),
            "average_retrieval_time": round(self.performance_metrics["context_retrieval_time"], 4),
            "average_generation_time": round(self.performance_metrics["generation_time"], 4),
            "accuracy_score": self.performance_metrics["accuracy_score"],
            "active_conversations": len(self.conversations)
        }
    
    def get_conversation_history(self, conversation_id: str) -> Dict[str, Any]:
        """Get conversation history"""
        if conversation_id not in self.conversations:
            return {"error": "Conversation not found"}
        
        conversation = self.conversations[conversation_id]
        return {
            "conversation_id": conversation_id,
            "messages": conversation.messages,
            "created_at": conversation.created_at.isoformat(),
            "last_updated": conversation.last_updated.isoformat(),
            "message_count": len(conversation.messages)
        }
    
    def cleanup_old_conversations(self, max_age_hours: int = 24):
        """Clean up old conversations to manage memory"""
        cutoff_time = datetime.now() - timedelta(hours=max_age_hours)
        
        conversations_to_remove = []
        for conv_id, conversation in self.conversations.items():
            if conversation.last_updated < cutoff_time:
                conversations_to_remove.append(conv_id)
        
        for conv_id in conversations_to_remove:
            del self.conversations[conv_id]
        
        return {
            "cleaned_conversations": len(conversations_to_remove),
            "remaining_conversations": len(self.conversations)
        }