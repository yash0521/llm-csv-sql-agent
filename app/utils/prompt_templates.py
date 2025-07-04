from typing import Dict, List, Any
import json

class PromptTemplates:
    """Advanced prompt templates optimized for 95% accuracy in SQL generation"""
    
    def __init__(self):
        self.sql_examples = self._load_sql_examples()
        self.query_patterns = self._load_query_patterns()
    
    def get_sql_generation_prompt(self, natural_query: str, table_name: str, 
                                 table_info: Dict[str, Any], sample_data: List[Dict]) -> str:
        """Generate optimized SQL generation prompt with few-shot examples"""
        
        return f"""You are an expert SQL generator. Convert natural language queries to precise SQL queries.

CRITICAL RULES:
1. ONLY generate SELECT statements
2. Use EXACT column names from the schema
3. Handle NULL values properly
4. Use appropriate SQL functions and syntax
5. Always use the table name: {table_name}
6. Return ONLY the SQL query, no explanations

TABLE SCHEMA:
{self._format_table_schema(table_info)}

SAMPLE DATA:
{self._format_sample_data(sample_data)}

EXAMPLE PATTERNS:
{self._get_relevant_examples(natural_query)}

QUERY TO CONVERT: "{natural_query}"

REQUIREMENTS:
- Use exact column names from schema
- Handle case sensitivity properly  
- Use appropriate WHERE clauses for filtering
- Use GROUP BY for aggregations
- Use ORDER BY for sorting
- Limit results when appropriate
- Handle date/time formats correctly

SQL QUERY:"""

    def get_rag_system_prompt(self) -> str:
        """System prompt for RAG chat agent"""
        return """You are an intelligent assistant with access to a knowledge base. 

Your capabilities:
- Answer questions using retrieved context
- Provide accurate, relevant information
- Cite sources when possible
- Handle follow-up questions naturally
- Maintain conversation context

Guidelines:
1. Use retrieved context to answer questions accurately
2. If context doesn't contain the answer, say so clearly
3. Provide specific, actionable information when possible
4. Maintain a helpful, professional tone
5. Ask clarifying questions when needed

Always base your answers on the provided context and clearly indicate when information is not available."""

    def get_query_routing_prompt(self, query: str) -> str:
        """Prompt for determining query complexity for model routing"""
        return f"""Analyze this query and determine its complexity level for optimal model routing.

QUERY: "{query}"

COMPLEXITY LEVELS:
- SIMPLE: Basic questions, definitions, simple facts (use cheaper model)
- MEDIUM: Analysis, comparisons, multi-step reasoning (use standard model)  
- COMPLEX: Advanced analysis, creative tasks, complex reasoning (use premium model)

FACTORS TO CONSIDER:
- Number of concepts involved
- Required reasoning depth
- Need for creativity or nuanced understanding
- Factual vs analytical nature
- Domain expertise required

Respond with ONLY: SIMPLE, MEDIUM, or COMPLEX"""

    def get_error_handling_prompt(self, error: str, sql_query: str, natural_query: str) -> str:
        """Prompt for handling and fixing SQL errors"""
        return f"""Fix this SQL query error:

ORIGINAL QUERY: "{natural_query}"
GENERATED SQL: {sql_query}
ERROR MESSAGE: {error}

COMMON ERROR FIXES:
- Column name misspellings
- Incorrect table references
- Missing quotes around strings
- Wrong aggregation functions
- Incorrect JOIN syntax
- Case sensitivity issues

Generate a corrected SQL query that fixes the error:"""

    def _format_table_schema(self, table_info: Dict[str, Any]) -> str:
        """Format table schema for prompt"""
        schema_lines = []
        for col in table_info['columns']:
            line = f"- {col['name']} ({col['type']})"
            if col['null_count'] > 0:
                line += f" [nullable]"
            if 'sample_values' in col:
                line += f" [examples: {', '.join(map(str, col['sample_values'][:3]))}]"
            schema_lines.append(line)
        
        return "\n".join(schema_lines)
    
    def _format_sample_data(self, sample_data: List[Dict]) -> str:
        """Format sample data for prompt"""
        if not sample_data:
            return "No sample data available"
        
        # Show first 2 rows as JSON for clarity
        formatted_data = []
        for i, row in enumerate(sample_data[:2]):
            formatted_data.append(f"Row {i+1}: {json.dumps(row, default=str)}")
        
        return "\n".join(formatted_data)
    
    def _get_relevant_examples(self, natural_query: str) -> str:
        """Get relevant SQL examples based on query type"""
        query_lower = natural_query.lower()
        examples = []
        
        # Pattern matching for different query types
        if any(word in query_lower for word in ['count', 'how many', 'number of']):
            examples.append(self.sql_examples['count'])
        
        if any(word in query_lower for word in ['average', 'mean', 'avg']):
            examples.append(self.sql_examples['average'])
            
        if any(word in query_lower for word in ['group', 'by', 'each']):
            examples.append(self.sql_examples['group_by'])
            
        if any(word in query_lower for word in ['top', 'highest', 'maximum', 'max']):
            examples.append(self.sql_examples['top_n'])
            
        if any(word in query_lower for word in ['where', 'filter', 'only', 'with']):
            examples.append(self.sql_examples['filter'])
        
        # Default examples if no specific pattern matches
        if not examples:
            examples = [self.sql_examples['basic'], self.sql_examples['filter']]
        
        return "\n\n".join(examples)
    
    def _load_sql_examples(self) -> Dict[str, str]:
        """Load pre-defined SQL examples for few-shot learning"""
        return {
            'basic': """Natural: "Show me all records"
SQL: SELECT * FROM table_name LIMIT 100""",
            
            'count': """Natural: "How many records are there?"
SQL: SELECT COUNT(*) as total_records FROM table_name""",
            
            'average': """Natural: "What is the average salary?"
SQL: SELECT AVG(salary) as average_salary FROM table_name""",
            
            'group_by': """Natural: "Count records by department"
SQL: SELECT department, COUNT(*) as count FROM table_name GROUP BY department""",
            
            'top_n': """Natural: "Show top 5 highest salaries"
SQL: SELECT * FROM table_name ORDER BY salary DESC LIMIT 5""",
            
            'filter': """Natural: "Show employees in sales department"
SQL: SELECT * FROM table_name WHERE department = 'Sales'""",
            
            'multiple_conditions': """Natural: "Show employees with salary > 50000 in IT department"
SQL: SELECT * FROM table_name WHERE salary > 50000 AND department = 'IT'""",
            
            'date_filter': """Natural: "Show records from last year"
SQL: SELECT * FROM table_name WHERE date_column >= DATE('now', '-1 year')""",
            
            'aggregation': """Natural: "Total sales by month"
SQL: SELECT strftime('%Y-%m', date_column) as month, SUM(sales) as total_sales 
FROM table_name GROUP BY strftime('%Y-%m', date_column) ORDER BY month"""
        }
    
    def _load_query_patterns(self) -> Dict[str, List[str]]:
        """Load common query patterns for better matching"""
        return {
            'aggregation': [
                'sum', 'total', 'average', 'mean', 'count', 'max', 'min', 
                'maximum', 'minimum', 'highest', 'lowest'
            ],
            'filtering': [
                'where', 'with', 'having', 'only', 'filter', 'exclude', 
                'include', 'containing', 'matching'
            ],
            'grouping': [
                'by', 'group', 'each', 'per', 'breakdown', 'category', 
                'categorize', 'segment'
            ],
            'sorting': [
                'order', 'sort', 'arrange', 'rank', 'top', 'bottom', 
                'first', 'last', 'ascending', 'descending'
            ],
            'limiting': [
                'top', 'first', 'last', 'limit', 'only', 'just', 
                'sample', 'few', 'some'
            ]
        }
    
    def get_validation_prompt(self, sql_query: str, natural_query: str, 
                            table_info: Dict[str, Any]) -> str:
        """Generate prompt for validating SQL query correctness"""
        return f"""Validate this SQL query for correctness and safety:

NATURAL QUERY: "{natural_query}"
GENERATED SQL: {sql_query}

TABLE COLUMNS: {[col['name'] for col in table_info['columns']]}

CHECK FOR:
1. Column names exist in table
2. Syntax is correct
3. Query matches the natural language intent
4. No dangerous operations (DROP, DELETE, etc.)
5. Proper data types and functions

VALIDATION RESULT (respond with VALID or list specific issues):"""

    def get_optimization_prompt(self, sql_query: str) -> str:
        """Generate prompt for SQL query optimization"""
        return f"""Optimize this SQL query for better performance:

ORIGINAL SQL: {sql_query}

OPTIMIZATION TECHNIQUES:
- Use appropriate indexes
- Avoid SELECT *
- Use LIMIT when appropriate  
- Optimize WHERE clauses
- Use efficient JOINs
- Consider subquery vs JOIN performance

OPTIMIZED SQL:"""

    def get_explanation_prompt(self, natural_query: str, sql_query: str) -> str:
        """Generate prompt for explaining SQL queries to users"""
        return f"""Explain this SQL query in simple, non-technical terms:

USER ASKED: "{natural_query}"
SQL GENERATED: {sql_query}

Explain:
1. What data is being retrieved
2. Any filtering or conditions applied
3. How results are organized/sorted
4. What the output will look like

Use simple language that a non-technical person can understand."""