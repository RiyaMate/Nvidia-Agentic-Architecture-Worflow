# agents/snowflake_agent.py
import os
import snowflake.connector
import pandas as pd
import matplotlib.pyplot as plt
import io
from io import BytesIO
import base64
from langchain.tools import Tool
from langchain.agents import AgentType, initialize_agent
from langchain_anthropic import ChatAnthropic
from dotenv import load_dotenv
import time
from typing import Dict, Any

load_dotenv(override=True)

# Initialize Snowflake connection
conn = snowflake.connector.connect(
    user=os.environ.get('SNOWFLAKE_USER'),
    password=os.environ.get('SNOWFLAKE_PASSWORD'),
    account=os.environ.get('SNOWFLAKE_ACCOUNT'),
    warehouse=os.environ.get('SNOWFLAKE_WAREHOUSE'),
    database=os.environ.get('SNOWFLAKE_DATABASE'),
    schema=os.environ.get('SNOWFLAKE_SCHEMA')
)
 
def query_snowflake(question: str) -> Dict:
    """Query Snowflake with predefined queries based on question intent"""
    # Default query for financial metrics
    base_query = "SELECT * FROM Valuation_Measures ORDER BY DATE DESC LIMIT 5"
    
    try:
        # Execute query and get DataFrame
        df = pd.read_sql(base_query, conn)
        
        # Generate summary
        summary = {
            "metrics": df.to_dict('records'),
            "latest_date": str(df['DATE'].max()),
            "query_status": "success"
        }
        
        return summary
        
    except Exception as e:
        return {
            "error": str(e),
            "query_status": "failed"
        }

 
def get_valuation_summary(query: str = None) -> dict:
    """Get NVIDIA valuation metrics visualization as a stacked area chart."""
    try:
        # Use base query
        df = pd.read_sql("SELECT * FROM Valuation_Measures ORDER BY DATE DESC LIMIT 5", conn)
        
        # Normalize the data for all metrics (excluding the DATE column)
        df_normalized = df.copy()
        metrics = df.columns[1:]  # Exclude the DATE column
        for metric in metrics:
            df_normalized[metric] = (df[metric] - df[metric].min()) / (df[metric].max() - df[metric].min())
        
        # Plot the stacked area chart with reduced figure size
        plt.figure(figsize=(8, 6))  # Reduced from (12, 8)
        x = df_normalized["DATE"].astype(str)
        y = df_normalized[metrics]
        plt.stackplot(x, y.T, labels=metrics, alpha=0.8)
        
        plt.xlabel("Date", fontsize=10)
        plt.ylabel("Normalized Value", fontsize=10)
        plt.title("NVIDIA Metrics Over Time", fontsize=12)
        plt.xticks(rotation=45, ha="right", fontsize=8)
        plt.legend(loc="upper left", title="Metrics", fontsize=8)
        plt.tight_layout()
        
        # Save chart with lower resolution
        chart_file_path = "nvidia_stacked_area_chart.png"
        plt.savefig(chart_file_path, format="png", dpi=150)  # Reduced from 300 dpi
        plt.close()
        
        # Create a simplified summary instead of using the full DataFrame
        summary_dict = {
            "dates": df_normalized["DATE"].astype(str).tolist(),
            "latest_values": df_normalized[metrics].iloc[0].to_dict(),
            "metrics_analyzed": len(metrics)
        }
        
        return {
            "summary": str(summary_dict),
            "status": "success"
        }
    
    except Exception as e:
        return {
            "error": str(e),
            "status": "failed"
        }

# Create LangChain tool for the Snowflake agent
snowflake_tool = Tool(
    name="nvidia_financial_metrics",
    description="Get NVIDIA financial valuation metrics from Snowflake",
    func=get_valuation_summary
)

llm = ChatAnthropic(
    model="claude-3-haiku-20240307",
    temperature=0,
    max_tokens_to_sample=150,  # Reduced from 300
    anthropic_api_key=os.environ.get('ANTHROPIC_API_KEY')  # Get from environment instead of hardcoding
) 

try:
    # Create agent with the tool
    # Simplify agent initialization
    agent = initialize_agent(
        tools=[snowflake_tool],
        llm=llm,
        agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION,  # Add specific agent type
        handle_parsing_errors=True,
        max_iterations=2,  # Limit iterations to reduce token usage
        early_stopping_method="generate"  # Add early stopping
    )
except Exception as e:
    print(f"Error initializing agent: {str(e)}")
    print("Available Claude models: claude-3-opus-20240229, claude-3-sonnet-20240229, claude-3-haiku-20240307")
    raise
 
def get_ai_analysis():
    """Get AI-generated analysis of NVIDIA metrics"""
    prompt = """Analyze NVIDIA's financial metrics and provide insights.
    Focus on key trends in market cap, PE ratios, and other valuation measures.
    Keep the analysis brief and highlight the most important changes."""
    
    try:
        # Get the metrics data first
        metrics_data = get_valuation_summary()
        if metrics_data["status"] == "failed":
            return f"Error getting metrics: {metrics_data['error']}"
        
        # Print debug information
        print("Retrieved metrics data:")
        print(metrics_data["summary"])
        
        # Invoke the agent with specific instructions
        response = agent.run(prompt)  # Using run() instead of invoke()
        
        return f"""
Analysis Results:
----------------
{response}

Note: A visualization has been saved as 'nvidia_stacked_area_chart.png'
"""
    except Exception as e:
        print(f"Error during analysis: {str(e)}")
        return "Analysis unavailable - Please try again later."

if __name__ == "__main__":
    print("Starting NVIDIA metrics analysis...")
    analysis = get_ai_analysis()
    print("\nAnalysis Output:")
    print("---------------")
    print(analysis)