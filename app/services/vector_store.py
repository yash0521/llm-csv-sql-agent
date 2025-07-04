import asyncio
import chromadb
import numpy as np
from typing import Dict, List, Any, Optional
import hashlib
import json
from datetime import datetime
from sentence_transformers import SentenceTransformer
import uuid

class VectorStore:
    """Advanced vector store for RAG with ChromaDB backend"""
    
    def __init__(self, collection_name: str = "rag_documents", 
                 embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"):
        self.collection_name = collection_name
        self.embedding_model_name = embedding_model
        self.client = None
        self.collection = None
        self.embedding_model = None
        self.document_cache = {}
        self.performance_metrics = {
            "total_documents": 0,
            "total_queries": 0,
            "average_query_time": 0,
            "cache_hits": 0
        }
        self._initialize()
    
    def _initialize(self):
        """Initialize ChromaDB client and embedding model"""
        try:
            # Initialize ChromaDB client (persistent storage)
            self.client = chromadb.PersistentClient(path="./chroma_db")
            
            # Get or create collection
            self.collection = self.client.get_or_create_collection(
                name=self.collection_name,
                metadata={"description": "RAG document store"}
            )
            
            # Initialize embedding model
            self.embedding_model = SentenceTransformer(self.embedding_model_name)
            
            print(f"✅ Vector store initialized with {self.collection.count()} documents")
            
        except Exception as e:
            print(f"❌ Failed to initialize vector store: {e}")
            # Fallback to in-memory ChromaDB
            self.client = chromadb.Client()
            self.collection = self.client.get_or_create_collection(self.collection_name)
            self.embedding_model = SentenceTransformer(self.embedding_model_name)
    
    async def add_documents(self, documents: List[Dict[str, Any]]) -> int:
        """Add documents to the vector store"""
        
        if not documents:
            return 0
        
        try:
            # Prepare data for ChromaDB
            texts = []
            metadatas = []
            ids = []
            
            for doc in documents:
                # Extract text content
                content = doc.get('content', doc.get('text', ''))
                if not content:
                    continue
                
                # Generate unique ID
                doc_id = doc.get('id', str(uuid.uuid4()))
                
                # Prepare metadata
                metadata = {
                    'source': doc.get('source', 'unknown'),
                    'title': doc.get('title', 'Untitled'),
                    'url': doc.get('url', ''),
                    'type': doc.get('type', 'document'),
                    'created_at': doc.get('created_at', datetime.now().isoformat()),
                    'length': len(content)
                }
                
                texts.append(content)
                metadatas.append(metadata)
                ids.append(doc_id)
            
            if not texts:
                return 0
            
            # Generate embeddings
            embeddings = await self._generate_embeddings(texts)
            
            # Add to collection
            self.collection.add(
                documents=texts,
                metadatas=metadatas,
                ids=ids,
                embeddings=embeddings.tolist()
            )
            
            # Update metrics
            self.performance_metrics["total_documents"] += len(texts)
            
            return len(texts)
            
        except Exception as e:
            print(f"Error adding documents: {e}")
            return 0
    
    async def similarity_search(self, query: str, k: int = 5, 
                              threshold: float = 0.7) -> List[Dict[str, Any]]:
        """Search for similar documents"""
        
        import time
        start_time = time.time()
        
        try:
            # Check cache first
            cache_key = self._generate_cache_key(query, k, threshold)
            if cache_key in self.document_cache:
                self.performance_metrics["cache_hits"] += 1
                return self.document_cache[cache_key]
            
            # Generate query embedding
            query_embedding = await self._generate_embeddings([query])
            
            # Search in ChromaDB
            results = self.collection.query(
                query_embeddings=query_embedding[0].tolist(),
                n_results=k,
                include=['documents', 'metadatas', 'distances']
            )
            
            # Process results
            documents = []
            for i, (doc, metadata, distance) in enumerate(zip(
                results['documents'][0],
                results['metadatas'][0],
                results['distances'][0]
            )):
                # Convert distance to similarity score (ChromaDB uses cosine distance)
                similarity = 1 - distance
                
                if similarity >= threshold:
                    documents.append({
                        'content': doc,
                        'metadata': metadata,
                        'score': similarity,
                        'rank': i + 1
                    })
            
            # Cache results
            self.document_cache[cache_key] = documents
            
            # Update metrics
            query_time = time.time() - start_time
            self._update_query_metrics(query_time)
            
            return documents
            
        except Exception as e:
            print(f"Error in similarity search: {e}")
            return []
    
    async def hybrid_search(self, query: str, k: int = 5, 
                          keyword_weight: float = 0.3) -> List[Dict[str, Any]]:
        """Hybrid search combining semantic and keyword search"""
        
        try:
            # Semantic search
            semantic_results = await self.similarity_search(query, k=k*2)
            
            # Keyword search (simple implementation)
            keyword_results = await self._keyword_search(query, k=k*2)
            
            # Combine and re-rank results
            combined_results = self._combine_search_results(
                semantic_results, keyword_results, keyword_weight
            )
            
            return combined_results[:k]
            
        except Exception as e:
            print(f"Error in hybrid search: {e}")
            return await self.similarity_search(query, k)
    
    async def _keyword_search(self, query: str, k: int = 10) -> List[Dict[str, Any]]:
        """Simple keyword search implementation"""
        
        try:
            # Get all documents
            all_docs = self.collection.get(include=['documents', 'metadatas'])
            
            if not all_docs['documents']:
                return []
            
            # Simple keyword matching
            query_words = set(query.lower().split())
            results = []
            
            for doc, metadata in zip(all_docs['documents'], all_docs['metadatas']):
                doc_words = set(doc.lower().split())
                
                # Calculate keyword overlap
                overlap = len(query_words.intersection(doc_words))
                if overlap > 0:
                    score = overlap / len(query_words.union(doc_words))
                    results.append({
                        'content': doc,
                        'metadata': metadata,
                        'score': score,
                        'rank': 0
                    })
            
            # Sort by score
            results.sort(key=lambda x: x['score'], reverse=True)
            
            # Add ranks
            for i, result in enumerate(results):
                result['rank'] = i + 1
            
            return results[:k]
            
        except Exception as e:
            print(f"Error in keyword search: {e}")
            return []
    
    def _combine_search_results(self, semantic_results: List[Dict], 
                              keyword_results: List[Dict], 
                              keyword_weight: float) -> List[Dict[str, Any]]:
        """Combine semantic and keyword search results"""
        
        # Create a map of content to results
        content_to_result = {}
        
        # Add semantic results
        for result in semantic_results:
            content = result['content']
            content_to_result[content] = {
                'content': content,
                'metadata': result['metadata'],
                'semantic_score': result['score'],
                'keyword_score': 0.0,
                'rank': result['rank']
            }
        
        # Add keyword results
        for result in keyword_results:
            content = result['content']
            if content in content_to_result:
                content_to_result[content]['keyword_score'] = result['score']
            else:
                content_to_result[content] = {
                    'content': content,
                    'metadata': result['metadata'],
                    'semantic_score': 0.0,
                    'keyword_score': result['score'],
                    'rank': result['rank']
                }
        
        # Calculate combined scores
        combined_results = []
        for result in content_to_result.values():
            combined_score = (
                result['semantic_score'] * (1 - keyword_weight) +
                result['keyword_score'] * keyword_weight
            )
            
            combined_results.append({
                'content': result['content'],
                'metadata': result['metadata'],
                'score': combined_score,
                'semantic_score': result['semantic_score'],
                'keyword_score': result['keyword_score'],
                'rank': 0
            })
        
        # Sort by combined score
        combined_results.sort(key=lambda x: x['score'], reverse=True)
        
        # Update ranks
        for i, result in enumerate(combined_results):
            result['rank'] = i + 1
        
        return combined_results
    
    async def _generate_embeddings(self, texts: List[str]) -> np.ndarray:
        """Generate embeddings for texts"""
        
        # Run embedding generation in thread pool to avoid blocking
        loop = asyncio.get_event_loop()
        embeddings = await loop.run_in_executor(
            None, self.embedding_model.encode, texts
        )
        
        return embeddings
    
    def _generate_cache_key(self, query: str, k: int, threshold: float) -> str:
        """Generate cache key for search results"""
        key_string = f"{query}|{k}|{threshold}"
        return hashlib.md5(key_string.encode()).hexdigest()
    
    def _update_query_metrics(self, query_time: float):
        """Update query performance metrics"""
        self.performance_metrics["total_queries"] += 1
        
        # Update average query time
        count = self.performance_metrics["total_queries"]
        current_avg = self.performance_metrics["average_query_time"]
        new_avg = ((current_avg * (count - 1)) + query_time) / count
        self.performance_metrics["average_query_time"] = new_avg
    
    async def get_document_count(self) -> int:
        """Get total number of documents in the store"""
        try:
            return self.collection.count()
        except Exception:
            return 0
    
    async def delete_documents(self, document_ids: List[str]) -> int:
        """Delete documents by IDs"""
        try:
            self.collection.delete(ids=document_ids)
            return len(document_ids)
        except Exception as e:
            print(f"Error deleting documents: {e}")
            return 0
    
    async def update_document(self, document_id: str, 
                            new_content: str, new_metadata: Dict = None) -> bool:
        """Update a document"""
        try:
            # Generate new embedding
            embedding = await self._generate_embeddings([new_content])
            
            # Update in ChromaDB
            self.collection.update(
                ids=[document_id],
                documents=[new_content],
                metadatas=[new_metadata or {}],
                embeddings=embedding[0].tolist()
            )
            
            return True
        except Exception as e:
            print(f"Error updating document: {e}")
            return False
    
    def get_collection_info(self) -> Dict[str, Any]:
        """Get information about the collection"""
        try:
            count = self.collection.count()
            
            # Get sample metadata
            sample = self.collection.peek(limit=5)
            
            return {
                "collection_name": self.collection_name,
                "document_count": count,
                "embedding_model": self.embedding_model_name,
                "sample_documents": len(sample.get('documents', [])),
                "performance_metrics": self.performance_metrics
            }
        except Exception as e:
            return {"error": str(e)}
    
    def clear_cache(self):
        """Clear the document cache"""
        self.document_cache.clear()
    
    async def rebuild_index(self):
        """Rebuild the entire index (useful for optimization)"""
        try:
            # Get all documents
            all_docs = self.collection.get(include=['documents', 'metadatas', 'ids'])
            
            if not all_docs['documents']:
                return True
            
            # Delete collection
            self.client.delete_collection(self.collection_name)
            
            # Recreate collection
            self.collection = self.client.create_collection(self.collection_name)
            
            # Re-add all documents
            embeddings = await self._generate_embeddings(all_docs['documents'])
            
            self.collection.add(
                documents=all_docs['documents'],
                metadatas=all_docs['metadatas'],
                ids=all_docs['ids'],
                embeddings=embeddings.tolist()
            )
            
            print(f"✅ Index rebuilt with {len(all_docs['documents'])} documents")
            return True
            
        except Exception as e:
            print(f"❌ Error rebuilding index: {e}")
            return False
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get performance statistics"""
        return {
            "total_documents": self.performance_metrics["total_documents"],
            "total_queries": self.performance_metrics["total_queries"],
            "average_query_time": round(self.performance_metrics["average_query_time"], 4),
            "cache_hit_rate": (
                self.performance_metrics["cache_hits"] / 
                max(self.performance_metrics["total_queries"], 1)
            ),
            "cache_size": len(self.document_cache),
            "embedding_model": self.embedding_model_name,
            "collection_name": self.collection_name
        }