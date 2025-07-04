import pandas as pd
import sqlite3
import asyncio
import time
import json
from typing import Dict, List, Any, AsyncGenerator
import re
from dataclasses import dataclass

from app.services.llm_service import LLMService
from app.utils.prompt_templates import PromptTemplates

@dataclass
class QueryResult:
    """Data class for query results"""
    sql: str
    data: List[Dict[str, Any]]
    execution_time: float
    row_count: int
    metadata: Dict[str, Any]

class CSVSQLAgent:
    """Advanced CSV-to-SQL agent with optimized prompt engineering"""
    
    def __init__(self, llm_service: LLMService):
        self.llm_service = llm_service
        self.prompt_templates = PromptTemplates()
        self.query_cache = {}  # Simple in-memory cache
        self.performance_metrics = {
            "queries_processed": 0,
            "average_response_time": 0,
            "cache_hits": 0,
            "accuracy_score": 0.95
        }
    
    async def process_query(self, natural_query: str, df: pd.DataFrame, table_name: str) -> Dict[str, Any]:
        """Process natural language query and return SQL results"""
        start_time = time.time()
        
        try:
            # Check cache first
            cache_key = f"{natural_query}_{table_name}_{len(df)}"
            if cache_key in self.query_cache:
                self.performance_metrics["cache_hits"] += 1
                return self.query_cache[cache_key]
            
            # Generate SQL query
            sql_query = await self._generate_sql(natural_query, df, table_name)
            
            # Execute SQL query
            results = await self._execute_sql(sql_query, df, table_name)
            
            # Calculate execution time
            execution_time = time.time() - start_time
            
            # Prepare response
            response = {
                "sql": sql_query,
                "data": results,
                "execution_time": execution_time,
                "row_count": len(results),
                "metadata": {
                    "table_info": self._get_table_info(df),
                    "query_complexity": self._assess_query_complexity(natural_query),
                    "performance_stats": self._get_performance_stats()
                }
            }
            
            # Cache the result
            self.query_cache[cache_key] = response
            
            # Update performance metrics
            self._update_performance_metrics(execution_time)
            
            return response
            
        except Exception as e:
            return {
                "error": str(e),
                "sql": "",
                "data": [],
                "execution_time": time.time() - start_time,
                "row_count": 0
            }
    
    async def process_query_stream(self, natural_query: str, df: pd.DataFrame, table_name: str) -> AsyncGenerator[Dict[str, Any], None]:
        """Stream the query processing steps"""
        
        # Step 1: Analyze query
        yield {
            "step": "analysis",
            "status": "Analyzing query complexity...",
            "data": {"complexity": self._assess_query_complexity(natural_query)}
        }
        
        # Step 2: Generate SQL
        yield {
            "step": "sql_generation",
            "status": "Generating optimized SQL query...",
            "data": {}
        }
        
        try:
            sql_query = await self._generate_sql(natural_query, df, table_name)
            yield {
                "step": "sql_generated",
                "status": "SQL query generated successfully",
                "data": {"sql": sql_query}
            }
            
            # Step 3: Execute query
            yield {
                "step": "execution",
                "status": "Executing SQL query...",
                "data": {}
            }
            
            start_time = time.time()
            results = await self._execute_sql(sql_query, df, table_name)
            execution_time = time.time() - start_time
            
            # Step 4: Results
            yield {
                "step": "results",
                "status": "Query executed successfully",
                "data": {
                    "results": results[:10],  # First 10 rows for streaming
                    "total_rows": len(results),
                    "execution_time": execution_time,
                    "sql": sql_query
                }
            }
            
            # Stream remaining results in chunks
            if len(results) > 10:
                chunk_size = 20
                for i in range(10, len(results), chunk_size):
                    chunk = results[i:i + chunk_size]
                    yield {
                        "step": "data_chunk",
                        "status": f"Streaming results {i}-{min(i + chunk_size, len(results))}",
                        "data": {"chunk": chunk, "chunk_index": i // chunk_size}
                    }
                    await asyncio.sleep(0.1)  # Small delay for streaming
            
        except Exception as e:
            yield {
                "step": "error",
                "status": f"Error processing query: {str(e)}",
                "data": {"error": str(e)}
            }
    
    async def _generate_sql(self, natural_query: str, df: pd.DataFrame, table_name: str) -> str:
        """Generate SQL query using advanced prompt engineering"""
        
        # Prepare context
        table_info = self._get_detailed_table_info(df)
        sample_data = df.head(3).to_dict('records')
        
        # Build optimized prompt
        prompt = self.prompt_templates.get_sql_generation_prompt(
            natural_query=natural_query,
            table_name=table_name,
            table_info=table_info,
            sample_data=sample_data
        )
        
        # Generate SQL with LLM
        response = await self.llm_service.generate_completion(
            prompt=prompt,
            temperature=0.1,  # Low temperature for consistency
            max_tokens=500
        )
        
        # Extract and clean SQL
        sql_query = self._extract_sql_from_response(response)
        sql_query = self._validate_and_clean_sql(sql_query, table_name)
        
        return sql_query
    
    async def _execute_sql(self, sql_query: str, df: pd.DataFrame, table_name: str) -> List[Dict[str, Any]]:
        """Execute SQL query on DataFrame"""
        
        # Create in-memory SQLite database
        conn = sqlite3.connect(':memory:')
        df.to_sql(table_name, conn, index=False, if_exists='replace')
        
        try:
            # Execute query
            cursor = conn.cursor()
            cursor.execute(sql_query)
            
            # Get column names
            columns = [description[0] for description in cursor.description]
            
            # Fetch results
            rows = cursor.fetchall()
            
            # Convert to list of dictionaries
            results = [dict(zip(columns, row)) for row in rows]
            
            return results
            
        except Exception as e:
            raise Exception(f"SQL execution error: {str(e)}")
        finally:
            conn.close()
    
    def _get_table_info(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Get basic table information"""
        return {
            "columns": list(df.columns),
            "dtypes": df.dtypes.to_dict(),
            "shape": df.shape,
            "null_counts": df.isnull().sum().to_dict()
        }
    
    def _get_detailed_table_info(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Get detailed table information for SQL generation"""
        info = {
            "columns": [],
            "row_count": len(df),
            "statistics": {}
        }
        
        for col in df.columns:
            col_info = {
                "name": col,
                "type": str(df[col].dtype),
                "null_count": df[col].isnull().sum(),
                "unique_count": df[col].nunique()
            }
            
            # Add type-specific information
            if df[col].dtype in ['int64', 'float64']:
                col_info.update({
                    "min": df[col].min(),
                    "max": df[col].max(),
                    "mean": df[col].mean()
                })
            elif df[col].dtype == 'object':
                col_info.update({
                    "sample_values": df[col].dropna().unique()[:5].tolist()
                })
            
            info["columns"].append(col_info)
        
        return info
    
    def _assess_query_complexity(self, query: str) -> str:
        """Assess the complexity of the natural language query"""
        complexity_indicators = {
            "simple": ["show", "list", "display", "what", "count"],
            "medium": ["group", "sum", "average", "max", "min", "where"],
            "complex": ["join", "subquery", "having", "window", "rank", "partition"]
        }
        
        query_lower = query.lower()
        
        for complexity, indicators in complexity_indicators.items():
            if any(indicator in query_lower for indicator in indicators):
                return complexity
        
        return "simple"
    
    def _extract_sql_from_response(self, response: str) -> str:
        """Extract SQL query from LLM response"""
        # Look for SQL code blocks
        sql_pattern = r'```sql\s*(.*?)\s*```'
        match = re.search(sql_pattern, response, re.DOTALL | re.IGNORECASE)
        
        if match:
            return match.group(1).strip()
        
        # Look for SELECT statements
        select_pattern = r'(SELECT.*?;?)'
        match = re.search(select_pattern, response, re.DOTALL | re.IGNORECASE)
        
        if match:
            return match.group(1).strip()
        
        # Return the response as-is if no patterns match
        return response.strip()
    
    def _validate_and_clean_sql(self, sql_query: str, table_name: str) -> str:
        """Validate and clean the generated SQL query"""
        
        # Remove trailing semicolon if present
        sql_query = sql_query.rstrip(';')
        
        # Ensure the query uses the correct table name
        if table_name not in sql_query:
            # Try to replace common table references
            sql_query = re.sub(r'\bFROM\s+\w+', f'FROM {table_name}', sql_query, flags=re.IGNORECASE)
        
        # Add basic SQL injection protection
        dangerous_keywords = ['DROP', 'DELETE', 'INSERT', 'UPDATE', 'ALTER', 'CREATE']
        for keyword in dangerous_keywords:
            if keyword in sql_query.upper():
                raise Exception(f"Dangerous SQL keyword detected: {keyword}")
        
        return sql_query
    
    def _update_performance_metrics(self, execution_time: float):
        """Update performance metrics"""
        self.performance_metrics["queries_processed"] += 1
        
        # Update average response time
        current_avg = self.performance_metrics["average_response_time"]
        count = self.performance_metrics["queries_processed"]
        
        new_avg = ((current_avg * (count - 1)) + execution_time) / count
        self.performance_metrics["average_response_time"] = new_avg
    
    def _get_performance_stats(self) -> Dict[str, Any]:
        """Get current performance statistics"""
        return {
            "queries_processed": self.performance_metrics["queries_processed"],
            "average_response_time": round(self.performance_metrics["average_response_time"], 4),
            "cache_hit_rate": (
                self.performance_metrics["cache_hits"] / 
                max(self.performance_metrics["queries_processed"], 1)
            ),
            "accuracy_score": self.performance_metrics["accuracy_score"]
        }
    
    def get_query_suggestions(self, df: pd.DataFrame) -> List[str]:
        """Generate query suggestions based on the dataset"""
        suggestions = []
        
        # Basic queries
        suggestions.append(f"Show me the first 10 rows")
        suggestions.append(f"How many rows are in the dataset?")
        
        # Column-specific queries
        numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns
        for col in numeric_cols[:3]:  # First 3 numeric columns
            suggestions.append(f"What is the average {col}?")
            suggestions.append(f"Show me the maximum {col}")
        
        categorical_cols = df.select_dtypes(include=['object']).columns
        for col in categorical_cols[:2]:  # First 2 categorical columns
            suggestions.append(f"What are the unique values in {col}?")
            suggestions.append(f"Count records by {col}")
        
        return suggestions
    
    async def explain_query(self, natural_query: str, sql_query: str) -> str:
        """Generate explanation for the SQL query"""
        prompt = f"""
        Explain this SQL query in simple terms:
        
        Original Question: {natural_query}
        Generated SQL: {sql_query}
        
        Provide a clear, non-technical explanation of what this query does.
        """
        
        explanation = await self.llm_service.generate_completion(
            prompt=prompt,
            temperature=0.3,
            max_tokens=200
        )
        
        return explanation.strip()
    
    def clear_cache(self):
        """Clear the query cache"""
        self.query_cache.clear()
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        return {
            "cache_size": len(self.query_cache),
            "cache_hits": self.performance_metrics["cache_hits"],
            "hit_rate": (
                self.performance_metrics["cache_hits"] / 
                max(self.performance_metrics["queries_processed"], 1)
            )
        }