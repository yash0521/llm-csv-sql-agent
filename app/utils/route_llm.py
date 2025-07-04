import re
import time
from typing import Dict, List, Any, Tuple
from dataclasses import dataclass
from enum import Enum

class ModelTier(Enum):
    """Model tiers for routing decisions"""
    CHEAP = "cheap"      # GPT-3.5-turbo, Claude Haiku
    STANDARD = "standard" # GPT-4, Claude Sonnet  
    PREMIUM = "premium"   # GPT-4-turbo, Claude Opus

@dataclass
class ModelConfig:
    """Configuration for each model"""
    name: str
    tier: ModelTier
    cost_per_1k_tokens: float
    max_tokens: int
    context_window: int
    speed_score: float  # Higher = faster
    quality_score: float  # Higher = better quality

@dataclass
class RouteDecision:
    """Routing decision with reasoning"""
    selected_model: str
    tier: ModelTier
    confidence: float
    reasoning: str
    estimated_cost: float
    complexity_score: float

class RouteLLM:
    """Advanced LLM router for cost and performance optimization"""
    
    def __init__(self):
        self.models = self._initialize_models()
        self.routing_stats = {
            "total_requests": 0,
            "cost_savings": 0.0,
            "tier_usage": {tier.value: 0 for tier in ModelTier},
            "accuracy_maintained": 0.95
        }
        self.complexity_cache = {}
    
    def route_query(self, query: str, context: str = "", user_preferences: Dict = None) -> str:
        """Route query to optimal model based on complexity and requirements"""
        
        # Analyze query complexity
        complexity_analysis = self._analyze_query_complexity(query, context)
        
        # Get user preferences
        preferences = user_preferences or {}
        cost_priority = preferences.get("cost_priority", 0.7)  # 0-1, higher = more cost conscious
        speed_priority = preferences.get("speed_priority", 0.5)
        quality_priority = preferences.get("quality_priority", 0.8)
        
        # Make routing decision
        decision = self._make_routing_decision(
            complexity_analysis, 
            cost_priority, 
            speed_priority, 
            quality_priority
        )
        
        # Update statistics
        self._update_routing_stats(decision)
        
        return decision.selected_model
    
    def route_with_details(self, query: str, context: str = "", 
                          user_preferences: Dict = None) -> RouteDecision:
        """Route query and return detailed decision information"""
        
        complexity_analysis = self._analyze_query_complexity(query, context)
        preferences = user_preferences or {}
        
        decision = self._make_routing_decision(
            complexity_analysis,
            preferences.get("cost_priority", 0.7),
            preferences.get("speed_priority", 0.5),
            preferences.get("quality_priority", 0.8)
        )
        
        self._update_routing_stats(decision)
        return decision
    
    def estimate_cost(self, query: str, model_name: str, estimated_response_tokens: int = 200) -> float:
        """Estimate cost for a query with specific model"""
        
        model = self._get_model_by_name(model_name)
        if not model:
            return 0.0
        
        # Estimate input tokens (rough approximation: 4 chars per token)
        input_tokens = len(query) // 4
        total_tokens = input_tokens + estimated_response_tokens
        
        return (total_tokens / 1000) * model.cost_per_1k_tokens
    
    def get_routing_recommendations(self, query: str) -> Dict[str, Any]:
        """Get routing recommendations for different scenarios"""
        
        complexity_analysis = self._analyze_query_complexity(query)
        
        recommendations = {}
        
        # Cost-optimized routing
        cost_decision = self._make_routing_decision(complexity_analysis, 0.9, 0.3, 0.6)
        recommendations["cost_optimized"] = {
            "model": cost_decision.selected_model,
            "estimated_cost": cost_decision.estimated_cost,
            "reasoning": "Prioritizes cost savings while maintaining quality"
        }
        
        # Performance-optimized routing  
        perf_decision = self._make_routing_decision(complexity_analysis, 0.2, 0.9, 0.8)
        recommendations["performance_optimized"] = {
            "model": perf_decision.selected_model,
            "estimated_cost": perf_decision.estimated_cost,
            "reasoning": "Prioritizes speed and quality over cost"
        }
        
        # Balanced routing
        balanced_decision = self._make_routing_decision(complexity_analysis, 0.6, 0.6, 0.7)
        recommendations["balanced"] = {
            "model": balanced_decision.selected_model,
            "estimated_cost": balanced_decision.estimated_cost,
            "reasoning": "Balances cost, speed, and quality"
        }
        
        return {
            "query_complexity": complexity_analysis,
            "recommendations": recommendations,
            "potential_savings": max(0, perf_decision.estimated_cost - cost_decision.estimated_cost)
        }
    
    def _initialize_models(self) -> Dict[str, ModelConfig]:
        """Initialize available models with their configurations"""
        return {
            "gpt-3.5-turbo": ModelConfig(
                name="gpt-3.5-turbo",
                tier=ModelTier.CHEAP,
                cost_per_1k_tokens=0.002,
                max_tokens=4096,
                context_window=16385,
                speed_score=0.9,
                quality_score=0.7
            ),
            "gpt-4": ModelConfig(
                name="gpt-4",
                tier=ModelTier.STANDARD,
                cost_per_1k_tokens=0.03,
                max_tokens=8192,
                context_window=8192,
                speed_score=0.6,
                quality_score=0.9
            ),
            "gpt-4-turbo": ModelConfig(
                name="gpt-4-turbo",
                tier=ModelTier.PREMIUM,
                cost_per_1k_tokens=0.01,
                max_tokens=4096,
                context_window=128000,
                speed_score=0.8,
                quality_score=0.95
            ),
            "claude-3-haiku": ModelConfig(
                name="claude-3-haiku",
                tier=ModelTier.CHEAP,
                cost_per_1k_tokens=0.00025,
                max_tokens=4096,
                context_window=200000,
                speed_score=0.95,
                quality_score=0.75
            ),
            "claude-3-sonnet": ModelConfig(
                name="claude-3-sonnet",
                tier=ModelTier.STANDARD,
                cost_per_1k_tokens=0.003,
                max_tokens=4096,
                context_window=200000,
                speed_score=0.7,
                quality_score=0.85
            ),
            "claude-3-opus": ModelConfig(
                name="claude-3-opus",
                tier=ModelTier.PREMIUM,
                cost_per_1k_tokens=0.015,
                max_tokens=4096,
                context_window=200000,
                speed_score=0.5,
                quality_score=0.98
            )
        }
    
    def _analyze_query_complexity(self, query: str, context: str = "") -> Dict[str, Any]:
        """Analyze query complexity using multiple factors"""
        
        # Check cache first
        cache_key = f"{query}:{len(context)}"
        if cache_key in self.complexity_cache:
            return self.complexity_cache[cache_key]
        
        complexity_factors = {
            "length_score": self._calculate_length_complexity(query),
            "semantic_score": self._calculate_semantic_complexity(query),
            "domain_score": self._calculate_domain_complexity(query),
            "reasoning_score": self._calculate_reasoning_complexity(query),
            "creativity_score": self._calculate_creativity_complexity(query),
            "context_score": self._calculate_context_complexity(context)
        }
        
        # Weighted overall complexity
        weights = {
            "length_score": 0.1,
            "semantic_score": 0.2,
            "domain_score": 0.15,
            "reasoning_score": 0.25,
            "creativity_score": 0.2,
            "context_score": 0.1
        }
        
        overall_complexity = sum(
            complexity_factors[factor] * weights[factor] 
            for factor in complexity_factors
        )
        
        analysis = {
            "overall_complexity": overall_complexity,
            "complexity_tier": self._get_complexity_tier(overall_complexity),
            "factors": complexity_factors,
            "explanation": self._explain_complexity(complexity_factors, overall_complexity)
        }
        
        # Cache the result
        self.complexity_cache[cache_key] = analysis
        return analysis
    
    def _calculate_length_complexity(self, query: str) -> float:
        """Calculate complexity based on query length"""
        word_count = len(query.split())
        if word_count < 10:
            return 0.2
        elif word_count < 25:
            return 0.5
        elif word_count < 50:
            return 0.7
        else:
            return 0.9
    
    def _calculate_semantic_complexity(self, query: str) -> float:
        """Calculate semantic complexity based on vocabulary and structure"""
        
        # Advanced vocabulary indicators
        advanced_words = [
            'analyze', 'synthesize', 'evaluate', 'critique', 'elaborate',
            'sophisticated', 'nuanced', 'comprehensive', 'intricate', 'complex'
        ]
        
        # Technical domain indicators
        technical_indicators = [
            'algorithm', 'optimization', 'methodology', 'framework', 'architecture',
            'implementation', 'analysis', 'statistical', 'mathematical', 'scientific'
        ]
        
        query_lower = query.lower()
        
        advanced_count = sum(1 for word in advanced_words if word in query_lower)
        technical_count = sum(1 for word in technical_indicators if word in query_lower)
        
        # Sentence complexity (multiple clauses, conjunctions)
        complex_structures = ['because', 'although', 'however', 'moreover', 'furthermore', 'whereas']
        structure_count = sum(1 for struct in complex_structures if struct in query_lower)
        
        semantic_score = min(1.0, (advanced_count * 0.2 + technical_count * 0.2 + structure_count * 0.3))
        return semantic_score
    
    def _calculate_domain_complexity(self, query: str) -> float:
        """Calculate complexity based on domain expertise required"""
        
        domain_keywords = {
            'high_complexity': [
                'quantum', 'molecular', 'genomic', 'neural networks', 'blockchain',
                'cryptography', 'bioinformatics', 'astrophysics', 'topology'
            ],
            'medium_complexity': [
                'programming', 'statistics', 'economics', 'biology', 'chemistry',
                'psychology', 'engineering', 'medicine', 'law', 'finance'
            ],
            'low_complexity': [
                'cooking', 'travel', 'sports', 'weather', 'basic math',
                'general knowledge', 'entertainment', 'shopping'
            ]
        }
        
        query_lower = query.lower()
        
        for complexity, keywords in domain_keywords.items():
            if any(keyword in query_lower for keyword in keywords):
                if complexity == 'high_complexity':
                    return 0.9
                elif complexity == 'medium_complexity':
                    return 0.6
                else:
                    return 0.3
        
        return 0.4  # Default medium-low complexity
    
    def _calculate_reasoning_complexity(self, query: str) -> float:
        """Calculate complexity based on reasoning requirements"""
        
        reasoning_indicators = {
            'high_reasoning': [
                'why', 'explain', 'analyze', 'compare', 'evaluate', 'critique',
                'pros and cons', 'implications', 'consequences', 'relationships'
            ],
            'medium_reasoning': [
                'how', 'what if', 'predict', 'estimate', 'calculate', 'solve',
                'recommend', 'suggest', 'optimize'
            ],
            'low_reasoning': [
                'what', 'when', 'where', 'who', 'list', 'show', 'tell', 'define'
            ]
        }
        
        query_lower = query.lower()
        
        for complexity, indicators in reasoning_indicators.items():
            if any(indicator in query_lower for indicator in indicators):
                if complexity == 'high_reasoning':
                    return 0.8
                elif complexity == 'medium_reasoning':
                    return 0.5
                else:
                    return 0.2
        
        return 0.3
    
    def _calculate_creativity_complexity(self, query: str) -> float:
        """Calculate complexity based on creativity requirements"""
        
        creative_indicators = [
            'creative', 'innovative', 'brainstorm', 'generate ideas', 'design',
            'write a story', 'compose', 'invent', 'imagine', 'artistic'
        ]
        
        analytical_indicators = [
            'factual', 'data', 'statistics', 'research', 'evidence',
            'citation', 'reference', 'study', 'report'
        ]
        
        query_lower = query.lower()
        
        creative_count = sum(1 for indicator in creative_indicators if indicator in query_lower)
        analytical_count = sum(1 for indicator in analytical_indicators if indicator in query_lower)
        
        if creative_count > analytical_count:
            return min(0.8, creative_count * 0.3)
        else:
            return min(0.4, analytical_count * 0.1)
    
    def _calculate_context_complexity(self, context: str) -> float:
        """Calculate complexity based on context length and content"""
        if not context:
            return 0.1
        
        context_length = len(context.split())
        
        if context_length < 100:
            return 0.2
        elif context_length < 500:
            return 0.4
        elif context_length < 1000:
            return 0.6
        else:
            return 0.8
    
    def _get_complexity_tier(self, complexity_score: float) -> str:
        """Convert complexity score to tier"""
        if complexity_score >= 0.7:
            return "high"
        elif complexity_score >= 0.4:
            return "medium"
        else:
            return "low"
    
    def _explain_complexity(self, factors: Dict[str, float], overall: float) -> str:
        """Generate explanation for complexity assessment"""
        explanations = []
        
        if factors["reasoning_score"] > 0.6:
            explanations.append("requires complex reasoning")
        if factors["domain_score"] > 0.7:
            explanations.append("involves specialized domain knowledge")
        if factors["creativity_score"] > 0.5:
            explanations.append("needs creative thinking")
        if factors["semantic_score"] > 0.6:
            explanations.append("uses advanced vocabulary")
        
        if not explanations:
            explanations.append("appears to be a straightforward query")
        
        tier = self._get_complexity_tier(overall)
        return f"Query classified as {tier} complexity because it {', '.join(explanations)}."
    
    def _make_routing_decision(self, complexity_analysis: Dict[str, Any], 
                             cost_priority: float, speed_priority: float, 
                             quality_priority: float) -> RouteDecision:
        """Make routing decision based on complexity and priorities"""
        
        complexity_score = complexity_analysis["overall_complexity"]
        complexity_tier = complexity_analysis["complexity_tier"]
        
        # Calculate scores for each model
        model_scores = {}
        
        for model_name, model_config in self.models.items():
            # Base score from model quality
            quality_score = model_config.quality_score
            
            # Adjust for complexity requirements
            if complexity_tier == "high" and model_config.tier == ModelTier.CHEAP:
                quality_score *= 0.7  # Penalize cheap models for complex queries
            elif complexity_tier == "low" and model_config.tier == ModelTier.PREMIUM:
                quality_score *= 0.9  # Slight penalty for overkill
            
            # Calculate weighted score
            cost_score = 1.0 / (model_config.cost_per_1k_tokens * 1000)  # Invert cost (higher = better)
            speed_score = model_config.speed_score
            
            final_score = (
                quality_score * quality_priority +
                cost_score * cost_priority +
                speed_score * speed_priority
            ) / (cost_priority + speed_priority + quality_priority)
            
            model_scores[model_name] = final_score
        
        # Select best model
        best_model_name = max(model_scores, key=model_scores.get)
        best_model = self.models[best_model_name]
        
        # Generate reasoning
        reasoning = self._generate_routing_reasoning(
            complexity_analysis, best_model, cost_priority, speed_priority, quality_priority
        )
        
        # Estimate cost
        estimated_cost = self.estimate_cost("dummy query", best_model_name)
        
        return RouteDecision(
            selected_model=best_model_name,
            tier=best_model.tier,
            confidence=model_scores[best_model_name],
            reasoning=reasoning,
            estimated_cost=estimated_cost,
            complexity_score=complexity_score
        )
    
    def _generate_routing_reasoning(self, complexity_analysis: Dict, model: ModelConfig,
                                  cost_priority: float, speed_priority: float, 
                                  quality_priority: float) -> str:
        """Generate human-readable reasoning for routing decision"""
        
        tier = complexity_analysis["complexity_tier"]
        explanation = complexity_analysis["explanation"]
        
        priorities = []
        if cost_priority > 0.7:
            priorities.append("cost efficiency")
        if speed_priority > 0.7:
            priorities.append("fast response")
        if quality_priority > 0.7:
            priorities.append("high quality")
        
        priority_text = ", ".join(priorities) if priorities else "balanced performance"
        
        return (f"Selected {model.name} ({model.tier.value} tier) for {tier} complexity query. "
                f"{explanation} Optimizing for {priority_text}.")
    
    def _get_model_by_name(self, model_name: str) -> ModelConfig:
        """Get model configuration by name"""
        return self.models.get(model_name)
    
    def _update_routing_stats(self, decision: RouteDecision):
        """Update routing statistics"""
        self.routing_stats["total_requests"] += 1
        self.routing_stats["tier_usage"][decision.tier.value] += 1
        
        # Estimate cost savings (compared to always using premium)
        premium_cost = max(model.cost_per_1k_tokens for model in self.models.values())
        actual_cost = self.models[decision.selected_model].cost_per_1k_tokens
        savings = premium_cost - actual_cost
        self.routing_stats["cost_savings"] += savings
    
    def get_routing_stats(self) -> Dict[str, Any]:
        """Get routing statistics"""
        total_requests = self.routing_stats["total_requests"]
        
        if total_requests == 0:
            return self.routing_stats
        
        # Calculate percentages
        tier_percentages = {
            tier: (count / total_requests) * 100 
            for tier, count in self.routing_stats["tier_usage"].items()
        }
        
        avg_savings_per_request = self.routing_stats["cost_savings"] / total_requests
        
        return {
            "total_requests": total_requests,
            "tier_usage_percentages": tier_percentages,
            "average_cost_savings_per_request": avg_savings_per_request,
            "total_cost_savings": self.routing_stats["cost_savings"],
            "accuracy_maintained": self.routing_stats["accuracy_maintained"],
            "cache_size": len(self.complexity_cache)
        }
    
    def clear_cache(self):
        """Clear the complexity analysis cache"""
        self.complexity_cache.clear()
    
    def optimize_for_budget(self, daily_budget: float, queries_per_day: int) -> Dict[str, Any]:
        """Provide recommendations for staying within budget"""
        
        cost_per_query_budget = daily_budget / queries_per_day
        
        recommendations = []
        
        # Find models within budget
        affordable_models = []
        for model_name, model in self.models.items():
            avg_cost = model.cost_per_1k_tokens * 0.3  # Assume 300 tokens average
            if avg_cost <= cost_per_query_budget:
                affordable_models.append((model_name, model, avg_cost))
        
        if not affordable_models:
            recommendations.append("Budget is too low for any available models")
        else:
            affordable_models.sort(key=lambda x: x[2])  # Sort by cost
            best_model = affordable_models[0]
            recommendations.append(f"Best model within budget: {best_model[0]} (${best_model[2]:.4f} per query)")
        
        return {
            "daily_budget": daily_budget,
            "queries_per_day": queries_per_day,
            "cost_per_query_budget": cost_per_query_budget,
            "affordable_models": [(m[0], m[2]) for m in affordable_models],
            "recommendations": recommendations
        }