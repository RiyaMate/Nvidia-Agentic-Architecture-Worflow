# agents/snowflake_agent.py
import os
import pandas as pd
import matplotlib.pyplot as plt
from sqlalchemy import create_engine
from langchain.tools import Tool
from langchain.agents import AgentType, initialize_agent
from langchain_anthropic import ChatAnthropic
from dotenv import load_dotenv
import warnings
from io import BytesIO
import base64
import time
from typing import Dict, Any
 
# Suppress deprecation warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
 
# Load environment variables
load_dotenv(override=True)
 
# Initialize Snowflake SQLAlchemy engine
engine = create_engine(
    f"snowflake://{os.environ.get('SNOWFLAKE_USER')}:{os.environ.get('SNOWFLAKE_PASSWORD')}@{os.environ.get('SNOWFLAKE_ACCOUNT')}/{os.environ.get('SNOWFLAKE_DATABASE')}/{os.environ.get('SNOWFLAKE_SCHEMA')}?warehouse={os.environ.get('SNOWFLAKE_WAREHOUSE')}"
)
 

 
def get_valuation_summary(query:str=None) -> dict:
    """Get NVIDIA valuation metrics visualization"""
    try:
        # Use base query
        df = pd.read_sql("SELECT * FROM Valuation_Measures ORDER BY DATE DESC LIMIT 5", engine)
        df.columns = df.columns.str.upper().str.strip()
       
        if df.empty:
            raise ValueError("No data returned from Snowflake. Ensure the table contains data.")
       
        # Generate visualization
        plt.figure(figsize=(10, 6))
        for date in df["DATE"].unique():
            subset = df[df["DATE"] == date]
            plt.bar(subset.columns[1:], subset.iloc[0, 1:], label=str(date))
       
        plt.xlabel("Metric")
        plt.ylabel("Value")
        plt.title("NVIDIA Valuation Metrics")
        plt.xticks(rotation=45)
        plt.legend()
       
        # Convert plot to base64
        buf = BytesIO()
        plt.savefig(buf, format='png', bbox_inches='tight')
        buf.seek(0)
        img_str = base64.b64encode(buf.getvalue()).decode()
       
        return {
            "chart": img_str,
            "summary": df.to_string(),
            "status": "success"
        }
       
    except Exception as e:
        return {
            "error": str(e),
            "status": "failed"
        }
 
def get_graph_specs_from_llm(data: pd.DataFrame) -> dict:
    """Get graph specifications from LLM based on the Snowflake data."""
    try:
        # Convert DataFrame to a JSON-like string for LLM input
        data_summary = data.to_dict(orient="list")
        prompt = f"""
        Based on the following NVIDIA financial data:
        {data_summary}
 
        Generate a graph specification in this format:
        - Title: [Graph title]
        - Type: [line/bar/scatter]
        - X-axis: [label and settings]
        - Y-axis: [label and settings]
        - Colors: [color scheme]
        - Additional elements: [grid, legend position, etc.]
 
        Focus on making the graph visually informative and easy to interpret.
        """
       
        # Send the prompt to the LLM
        response = llm.invoke(prompt)
        if not response or not hasattr(response, "content"):
            raise ValueError("LLM did not return a valid response.")
       
        # Extract the content from the AIMessage object
        if hasattr(response, "content"):
            response_text = response.content
        else:
            response_text = str(response)
       
        # Parse the LLM response into a dictionary
        specs = {}
        for line in response_text.split("\n"):
            if ":" in line:
                key, value = line.split(":", 1)
                specs[key.strip()] = value.strip()
       
        return specs
    except Exception as e:
        print(f"Error getting graph specs from LLM: {str(e)}")
        return {}
 
def create_graph_from_llm_specs(data: pd.DataFrame, specs: dict) -> str:
    """Create a graph based on LLM specifications."""
    try:
        if 'DATE' not in data.columns:
            raise ValueError("The 'DATE' column is missing from the data.")
       
        plt.figure(figsize=(12, 6))
        x = pd.to_datetime(data["DATE"]).dt.strftime('%Y-%m-%d')
       
        # Determine the graph type
        graph_type = specs.get("Type", "line").lower()
        y_label = specs.get("Y-axis", "Value")
        title = specs.get("Title", "Generated Graph")
       
        # Plot the graph based on the type
        if graph_type == "line":
            for col in data.columns[1:]:
                plt.plot(x, data[col], label=col, marker="o")
        elif graph_type == "bar":
            for col in data.columns[1:]:
                plt.bar(x, data[col], label=col, alpha=0.7)
        elif graph_type == "scatter":
            for col in data.columns[1:]:
                plt.scatter(x, data[col], label=col)
       
        # Apply additional formatting
        plt.title(title)
        plt.xlabel(specs.get("X-axis", "Date"))
        plt.ylabel(y_label)
        if "grid" in specs.get("Additional elements", "").lower():
            plt.grid(True, alpha=0.3)
        if "legend" in specs.get("Additional elements", "").lower():
            plt.legend(loc="upper left", bbox_to_anchor=(1.05, 1))
       
        plt.xticks(rotation=45, ha="right")
        plt.tight_layout()
       
        # Save the graph to a file
        chart_file_path = "llm_generated_graph.png"
        plt.savefig(chart_file_path, format="png", dpi=150, bbox_inches="tight")
        plt.close()
       
        return chart_file_path
    except Exception as e:
        print(f"Error creating graph: {str(e)}")
        return None
 
def get_valuation_summary_with_llm_graph() -> dict:
    """Get NVIDIA valuation metrics and generate a graph using LLM."""
    try:
        # Fetch data from Snowflake
        df = pd.read_sql("SELECT * FROM Valuation_Measures ORDER BY DATE ASC LIMIT 10", engine)
       
        # Normalize column names
        df.columns = df.columns.str.upper().str.strip()
 
        # Debugging
        print("DEBUG: Normalized DataFrame columns:", df.columns)
        print("DEBUG: First few rows of the DataFrame:\n", df.head())
       
        if df.empty:
            raise ValueError("No data returned from Snowflake. Ensure the table contains data.")
       
        # Ensure the DATE column is parsed as datetime
        if 'DATE' in df.columns:
            df['DATE'] = pd.to_datetime(df['DATE'], errors='coerce')
            if df['DATE'].isnull().all():
                raise ValueError("The 'DATE' column could not be parsed as datetime.")
        else:
            raise ValueError("The 'DATE' column is missing from the data.")
       
        # Get graph specifications from LLM
        graph_specs = get_graph_specs_from_llm(df)
        if not graph_specs:
            print("LLM failed to generate graph specifications. Using default graph settings.")
            graph_specs = {
                "Title": "Default Graph",
                "Type": "line",
                "X-axis": "Date",
                "Y-axis": "Value",
                "Additional elements": "grid, legend"
            }
       
        # Create the graph based on LLM specifications
        chart_file_path = create_graph_from_llm_specs(df, graph_specs)
        if not chart_file_path:
            raise ValueError("Failed to create graph from LLM specifications.")
       
        # Return the summary and graph path
        return {
            "summary": df.to_dict(orient="records"),
            "chart_path": chart_file_path,
            "graph_specs": graph_specs,
            "status": "success"
        }
    except Exception as e:
        print(f"Error in get_valuation_summary_with_llm_graph: {str(e)}")
        return {
            "error": str(e),
            "status": "failed"
        }
 
def get_ai_analysis_with_graph(prompt: str) -> str:
    """Get AI-generated analysis of NVIDIA metrics with LLM-generated graph."""
    try:
        # Get the valuation summary and graph
        result = get_valuation_summary_with_llm_graph()
        if result["status"] == "failed":
            return f"Error: {result['error']}"
       
        # Create a prompt for the LLM to analyze the data and graph
        prompt = f"""
        Analyze the following NVIDIA financial data and the generated graph:
        Data Summary:
        {result['summary']}
       
        Graph Specifications:
        {result['graph_specs']}
       
        Provide insights based on the data and the graph. Highlight key trends, patterns, and any significant observations.
        """
       
        # Get the analysis from the LLM
        response = llm.invoke(prompt)
       
        # Handle the AIMessage response correctly
        if hasattr(response, 'content'):
            analysis = response.content
        else:
            analysis = str(response)
       
        return f"""
Analysis Results:
----------------
{analysis}
 
Graph Path:
-----------
{result['chart_path']}
"""
    except Exception as e:
        print(f"Error during AI analysis with graph: {str(e)}")
        return "Analysis unavailable - Please try again later."
 
# Create LangChain tool for the Snowflake agent
snowflake_tool = Tool(
    name="nvidia_financial_metrics",
    description="Get NVIDIA financial valuation metrics from Snowflake",
    func=get_valuation_summary
)
 
llm = ChatAnthropic(
    model="claude-3-haiku-20240307",  
    temperature=0,
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
 
def get_ai_analysis(prompt: str) -> str:
    """Get AI-generated analysis of NVIDIA metrics"""
    try:
        response = agent.invoke({"input": prompt})
        return response.get("output", str(response))
    except Exception as e:
        print(f"Error during analysis: {str(e)}")
        return "Analysis unavailable - Rate limit exceeded. Please try again later."  
 
if __name__ == "__main__":
    analysis = get_ai_analysis_with_graph("Analyze NVIDIA financial metrics using the nvidia_financial_metrics tool.Provide a brief summary of key insights.")
    print(analysis)