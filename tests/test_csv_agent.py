import pytest
import pandas as pd
from app.agents.csv_sql_agent import CSVSQLAgent
from app.services.llm_service import LLMService

@pytest.fixture
def sample_df():
    return pd.DataFrame({
        'id': [1, 2, 3],
        'name': ['Alice', 'Bob', 'Charlie'],
        'salary': [50000, 60000, 70000]
    })

@pytest.mark.asyncio
async def test_csv_agent_initialization():
    llm_service = LLMService()
    agent = CSVSQLAgent(llm_service)
    assert agent is not None