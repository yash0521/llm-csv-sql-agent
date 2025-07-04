import asyncio
import openai
import time
from typing import Dict, List, Any, AsyncGenerator, Optional
import json
import os
from dataclasses import dataclass

@dataclass
class CompletionRequest:
    """Request configuration for LLM completion"""
    prompt: str
    model: str = "gpt-3.5-turbo"
    temperature: float = 0.7
    max_tokens: int = 1000
    stream: bool = False
    stop: Optional[List[str]] = None

@dataclass
class CompletionResponse:
    """Response from LLM completion"""
    text: str
    model: str
    usage: Dict[str, int]
    response_time: float
    finish_reason: str

class LLMService:
    """Service for managing multiple LLM providers with optimization"""
    
    def __init__(self):
        self.openai_client = None
        self.anthropic_client = None
        self._initialize_clients()
        self.request_cache = {}
        self.performance_metrics = {
            "total_requests": 0,
            "total_tokens": 0,
            "average_response_time": 0,
            "cache_hits": 0,
            "error_count": 0,
            "provider_usage": {}
        }
    
    def _initialize_clients(self):
        """Initialize API clients for different providers"""
        try:
            # OpenAI client
            openai_api_key = os.getenv("OPENAI_API_KEY")
            if openai_api_key:
                self.openai_client = openai.AsyncOpenAI(api_key=openai_api_key)
            
            # Add other providers as needed
            # self.anthropic_client = anthropic.AsyncAnthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
            
        except Exception as e:
            print(f"Warning: Failed to initialize some LLM clients: {e}")
    
    async def generate_completion(self, prompt: str, model: str = "gpt-3.5-turbo",
                                temperature: float = 0.7, max_tokens: int = 1000,
                                **kwargs) -> str:
        """Generate completion from specified model"""
        
        request = CompletionRequest(
            prompt=prompt,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            **kwargs
        )
        
        # Check cache first
        cache_key = self._generate_cache_key(request)
        if cache_key in self.request_cache:
            self.performance_metrics["cache_hits"] += 1
            return self.request_cache[cache_key].text
        
        start_time = time.time()
        
        try:
            response = await self._call_model(request)
            
            # Cache successful responses
            self.request_cache[cache_key] = response
            
            # Update metrics
            self._update_metrics(response, time.time() - start_time)
            
            return response.text
            
        except Exception as e:
            self.performance_metrics["error_count"] += 1
            raise Exception(f"LLM generation failed: {str(e)}")
    
    async def generate_completion_stream(self, prompt: str, model: str = "gpt-3.5-turbo",
                                       temperature: float = 0.7, max_tokens: int = 1000,
                                       **kwargs) -> AsyncGenerator[str, None]:
        """Generate streaming completion"""
        
        request = CompletionRequest(
            prompt=prompt,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            stream=True,
            **kwargs
        )
        
        try:
            async for chunk in self._stream_model(request):
                yield chunk
                await asyncio.sleep(0.01)  # Small delay for better streaming experience
                
        except Exception as e:
            self.performance_metrics["error_count"] += 1
            yield f"Error: {str(e)}"
    
    async def _call_model(self, request: CompletionRequest) -> CompletionResponse:
        """Call the appropriate model based on the request"""
        
        provider = self._get_provider_for_model(request.model)
        
        if provider == "openai":
            return await self._call_openai(request)
        elif provider == "anthropic":
            return await self._call_anthropic(request)
        else:
            raise Exception(f"Unsupported model: {request.model}")
    
    async def _stream_model(self, request: CompletionRequest) -> AsyncGenerator[str, None]:
        """Stream from the appropriate model"""
        
        provider = self._get_provider_for_model(request.model)
        
        if provider == "openai":
            async for chunk in self._stream_openai(request):
                yield chunk
        elif provider == "anthropic":
            async for chunk in self._stream_anthropic(request):
                yield chunk
        else:
            raise Exception(f"Unsupported model for streaming: {request.model}")
    
    async def _call_openai(self, request: CompletionRequest) -> CompletionResponse:
        """Call OpenAI API"""
        
        if not self.openai_client:
            raise Exception("OpenAI client not initialized")
        
        start_time = time.time()
        
        # Prepare messages format
        messages = [{"role": "user", "content": request.prompt}]
        
        response = await self.openai_client.chat.completions.create(
            model=request.model,
            messages=messages,
            temperature=request.temperature,
            max_tokens=request.max_tokens,
            stop=request.stop
        )
        
        response_time = time.time() - start_time
        
        return CompletionResponse(
            text=response.choices[0].message.content,
            model=request.model,
            usage={
                "prompt_tokens": response.usage.prompt_tokens,
                "completion_tokens": response.usage.completion_tokens,
                "total_tokens": response.usage.total_tokens
            },
            response_time=response_time,
            finish_reason=response.choices[0].finish_reason
        )
    
    async def _stream_openai(self, request: CompletionRequest) -> AsyncGenerator[str, None]:
        """Stream from OpenAI API"""
        
        if not self.openai_client:
            raise Exception("OpenAI client not initialized")
        
        messages = [{"role": "user", "content": request.prompt}]
        
        stream = await self.openai_client.chat.completions.create(
            model=request.model,
            messages=messages,
            temperature=request.temperature,
            max_tokens=request.max_tokens,
            stream=True
        )
        
        async for chunk in stream:
            if chunk.choices[0].delta.content:
                yield chunk.choices[0].delta.content
    
    async def _call_anthropic(self, request: CompletionRequest) -> CompletionResponse:
        """Call Anthropic API (placeholder)"""
        # Implement Anthropic API calls here
        raise Exception("Anthropic integration not implemented yet")
    
    async def _stream_anthropic(self, request: CompletionRequest) -> AsyncGenerator[str, None]:
        """Stream from Anthropic API (placeholder)"""
        # Implement Anthropic streaming here
        raise Exception("Anthropic streaming not implemented yet")
        yield ""  # This line will never be reached, but needed for generator
    
    def _get_provider_for_model(self, model: str) -> str:
        """Determine which provider to use for a given model"""
        
        openai_models = [
            "gpt-3.5-turbo", "gpt-4", "gpt-4-turbo", "gpt-4-turbo-preview",
            "gpt-3.5-turbo-16k", "gpt-4-32k"
        ]
        
        anthropic_models = [
            "claude-3-haiku", "claude-3-sonnet", "claude-3-opus",
            "claude-2", "claude-instant"
        ]
        
        if model in openai_models:
            return "openai"
        elif model in anthropic_models:
            return "anthropic"
        else:
            # Default to OpenAI for unknown models
            return "openai"
    
    def _generate_cache_key(self, request: CompletionRequest) -> str:
        """Generate cache key for request"""
        # Create a hash of the important parameters
        key_parts = [
            request.prompt,
            request.model,
            str(request.temperature),
            str(request.max_tokens)
        ]
        return "|".join(key_parts)
    
    def _update_metrics(self, response: CompletionResponse, request_time: float):
        """Update performance metrics"""
        
        self.performance_metrics["total_requests"] += 1
        self.performance_metrics["total_tokens"] += response.usage["total_tokens"]
        
        # Update average response time
        count = self.performance_metrics["total_requests"]
        current_avg = self.performance_metrics["average_response_time"]
        new_avg = ((current_avg * (count - 1)) + request_time) / count
        self.performance_metrics["average_response_time"] = new_avg
        
        # Update provider usage
        provider = self._get_provider_for_model(response.model)
        if provider not in self.performance_metrics["provider_usage"]:
            self.performance_metrics["provider_usage"][provider] = 0
        self.performance_metrics["provider_usage"][provider] += 1
    
    async def get_available_models(self) -> Dict[str, List[str]]:
        """Get list of available models by provider"""
        
        available_models = {
            "openai": [],
            "anthropic": []
        }
        
        if self.openai_client:
            try:
                # In a real implementation, you might query the API for available models
                available_models["openai"] = [
                    "gpt-3.5-turbo", "gpt-4", "gpt-4-turbo"
                ]
            except Exception as e:
                print(f"Failed to get OpenAI models: {e}")
        
        # Add other providers similarly
        
        return available_models
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get current performance statistics"""
        
        total_requests = self.performance_metrics["total_requests"]
        
        return {
            "total_requests": total_requests,
            "total_tokens": self.performance_metrics["total_tokens"],
            "average_response_time": round(self.performance_metrics["average_response_time"], 4),
            "cache_hit_rate": (
                self.performance_metrics["cache_hits"] / max(total_requests, 1)
            ),
            "error_rate": (
                self.performance_metrics["error_count"] / max(total_requests, 1)
            ),
            "provider_usage": self.performance_metrics["provider_usage"],
            "cache_size": len(self.request_cache)
        }
    
    async def test_model_availability(self, model: str) -> Dict[str, Any]:
        """Test if a specific model is available and working"""
        
        test_prompt = "Hello, this is a test. Please respond with 'Test successful.'"
        
        try:
            start_time = time.time()
            response = await self.generate_completion(
                prompt=test_prompt,
                model=model,
                max_tokens=50,
                temperature=0
            )
            response_time = time.time() - start_time
            
            return {
                "model": model,
                "available": True,
                "response_time": response_time,
                "response": response,
                "provider": self._get_provider_for_model(model)
            }
            
        except Exception as e:
            return {
                "model": model,
                "available": False,
                "error": str(e),
                "provider": self._get_provider_for_model(model)
            }
    
    def clear_cache(self):
        """Clear the request cache"""
        self.request_cache.clear()
    
    def set_cache_size_limit(self, limit: int):
        """Set maximum cache size"""
        if len(self.request_cache) > limit:
            # Remove oldest entries (simple FIFO)
            items_to_remove = len(self.request_cache) - limit
            keys_to_remove = list(self.request_cache.keys())[:items_to_remove]
            for key in keys_to_remove:
                del self.request_cache[key]
    
    async def batch_generate(self, prompts: List[str], model: str = "gpt-3.5-turbo",
                           **kwargs) -> List[str]:
        """Generate completions for multiple prompts concurrently"""
        
        tasks = [
            self.generate_completion(prompt=prompt, model=model, **kwargs)
            for prompt in prompts
        ]
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Convert exceptions to error messages
        processed_results = []
        for result in results:
            if isinstance(result, Exception):
                processed_results.append(f"Error: {str(result)}")
            else:
                processed_results.append(result)
        
        return processed_results
    
    def estimate_tokens(self, text: str) -> int:
        """Rough estimation of token count (4 chars per token average)"""
        return len(text) // 4
    
    def estimate_cost(self, prompt: str, model: str, estimated_response_tokens: int = 200) -> float:
        """Estimate cost for a completion request"""
        
        # Token costs per 1K tokens (approximate)
        costs = {
            "gpt-3.5-turbo": 0.002,
            "gpt-4": 0.03,
            "gpt-4-turbo": 0.01,
            "claude-3-haiku": 0.00025,
            "claude-3-sonnet": 0.003,
            "claude-3-opus": 0.015
        }
        
        cost_per_1k = costs.get(model, 0.002)  # Default to GPT-3.5 pricing
        
        prompt_tokens = self.estimate_tokens(prompt)
        total_tokens = prompt_tokens + estimated_response_tokens
        
        return (total_tokens / 1000) * cost_per_1k