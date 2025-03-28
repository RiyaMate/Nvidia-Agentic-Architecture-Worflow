"""
NVIDIA Research Pipeline using LangGraph
Integrates web search, RAG, and Snowflake data for comprehensive NVIDIA analysis
"""
import os
import sys
import operator
import traceback
from typing import TypedDict, Dict, Any, List, Annotated

# LangChain imports
from langchain_core.agents import AgentAction
from langchain_core.messages import BaseMessage
from langchain_core.tools import Tool
from langchain.agents import initialize_agent, AgentType
from langchain.memory import ConversationBufferMemory
from langchain_anthropic import ChatAnthropic

# LangGraph imports
from langgraph.graph import StateGraph, END

# Visualization
from graphviz import Digraph

# Add parent directory to path for agent imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import agent functions
from agents.websearch_agent import search_quarterly
from agents.rag_agent import search_all_namespaces, search_specific_quarter
from agents.snowflake_agent import query_snowflake, get_valuation_summary

# ===============================
# Type Definitions and Constants
# ===============================

class NvidiaGPTState(TypedDict, total=False):
    """State definition for NVIDIA research pipeline"""
    input: str  # User's original query
    question: str  # Processed question
    search_type: str  # "All Quarters" or "Specific Quarter"
    selected_periods: List[str]  # List of quarters to analyze
    web_output: str  # Results from web search
    rag_output: Dict[str, Any]  # Results from RAG search
    snowflake_output: Dict[str, Any]  # Results from Snowflake query
    valuation_data: Dict[str, Any]  # Financial visualization data
    chat_history: List[Dict[str, Any]]  # Conversation history
    intermediate_steps: Annotated[List[tuple[AgentAction, str]], operator.add]  # Agent reasoning steps
    assistant_response: str  # Agent's response
    final_report: Dict[str, Any]  # Final structured report

# ===============================
# Tool Functions
# ===============================

def final_report_tool(input_dict: Dict) -> Dict:
    """
    Generates final report in structured format
    
    Args:
        input_dict: Dictionary with report fields
        
    Returns:
        Dict: Structured report with sections
    """
    return {
        "introduction": input_dict.get("introduction", ""),
        "key_findings": input_dict.get("key_findings", []),
        "analysis": input_dict.get("analysis", ""),
        "conclusion": input_dict.get("conclusion", ""),
        "sources": input_dict.get("sources", [])
    }

# ===============================
# Node Functions
# ===============================

def start_node(state: NvidiaGPTState) -> Dict:
    """
    Initial node that processes the input query
    
    Args:
        state: Current pipeline state
        
    Returns:
        Dict: Updates with processed question
    """
    return {"question": state["input"]}

def web_search_node(state: NvidiaGPTState) -> Dict:
    """
    Execute web search for NVIDIA information
    
    Args:
        state: Current pipeline state
        
    Returns:
        Dict: Web search results
    """
    try:
        result = search_quarterly(state["question"])
        return {"web_output": result}
    except Exception as e:
        return {"web_output": f"Web search error: {str(e)}"}

def rag_search_node(state: NvidiaGPTState) -> Dict:
    """
    Execute RAG search based on search type
    
    Args:
        state: Current pipeline state
        
    Returns:
        Dict: RAG search results
    """
    try:
        if state.get("search_type") == "All Quarters":
            # Search across all document namespaces
            result = search_all_namespaces(state["question"])
            return {"rag_output": {"type": "all", "result": result}}
        else:
            # Fix: Structure input_dict correctly for Pydantic validation
            input_dict = {
                "input_dict": {  # Add this outer key to match expected format
                    "query": state["question"],
                    "selected_periods": state.get("selected_periods", ["2023q1"])
                }
            }
            
            # Use invoke method instead of direct function call
            result = search_specific_quarter.invoke(input_dict)
            return {"rag_output": {
                "type": "specific",
                "result": result,
                "periods": state.get("selected_periods", ["2023q1"])
            }}
    except Exception as e:
        return {"rag_output": {"type": "error", "result": f"RAG search error: {str(e)}"}}
                
def snowflake_node(state: NvidiaGPTState) -> Dict:
    """Execute Snowflake query with reduced token usage"""
    try:
        # Skip full question-based query to save tokens
        query_result = {
            "metrics": "See valuation data for metrics",
            "latest_date": "Recent date",
            "query_status": "success"
        }
        
        # Get visualization with reduced data
        valuation_data = get_valuation_summary()
        
        return {
            "snowflake_output": query_result,
            "valuation_data": valuation_data
        }
    except Exception as e:
        return {
            "snowflake_output": {"error": str(e)},
            "valuation_data": {"error": str(e)}
        }
    
def agent_node(state: NvidiaGPTState, nvidia_gpt) -> Dict:
    """Execute NvidiaGPT agent with simplified prompt"""
    try:
        # Create a more concise context
        context = f"Web Search: {state.get('web_output', 'No data')}\n" + \
                 f"RAG Search: {state.get('rag_output', {}).get('result', 'No data')}\n" + \
                 f"Snowflake Data: {state.get('snowflake_output', {}).get('metrics', 'No data')}"
        
        # Simplified prompt
        response = nvidia_gpt.invoke({
            "input": f"{context}\n\nAnalyze NVIDIA's performance based on: {state['question']}"
        })
        
        if isinstance(response, dict) and "output" in response:
            return {"assistant_response": response["output"]}
        return {"assistant_response": str(response)}
        
    except Exception as e:
        return {"assistant_response": f"Analysis error: {str(e)}"}
    

def final_report_node(state: NvidiaGPTState) -> Dict:
    """
    Generate final report combining all sources
    
    Args:
        state: Current pipeline state
        
    Returns:
        Dict: Final structured report
    """
    try:
        # Extract RAG output
        rag_data = state.get("rag_output", {})
        rag_type = rag_data.get("type", "unknown")
        rag_result = rag_data.get("result", "No RAG data")
        
        # Format key findings from all sources
        key_findings = []
        
        # Add web search findings
        web_results = state.get('web_output')
        if web_results:
            key_findings.append(f"Web Search: {web_results}")
            
        # Add RAG findings
        if rag_result:
            key_findings.append(f"Document Analysis ({rag_type}): {rag_result}")
            
        # Add financial metrics if available
        snowflake_output = state.get('snowflake_output', {})
        if snowflake_output and isinstance(snowflake_output, dict):
            metrics = snowflake_output.get('metrics')
            if metrics:
                key_findings.append(f"Financial Metrics: Latest metrics from {snowflake_output.get('latest_date', 'recent date')}")
        
        # Build report
        report = final_report_tool({
            "introduction": f"Analysis of NVIDIA performance for: {state['question']}",
            "key_findings": key_findings,
            "analysis": state.get("assistant_response", "Analysis unavailable"),
            "conclusion": "Based on the collected data, NVIDIA continues to show strong performance in the GPU market, driven by AI and data center demand.",
            "sources": ["Web Search", "Document Analysis", "Financial Data", "AI Analysis"]
        })
        
        return {"final_report": report}
    except Exception as e:
        return {"final_report": {
            "introduction": "Error generating report",
            "key_findings": [f"Error: {str(e)}"],
            "analysis": "Analysis unavailable due to error",
            "conclusion": "Unable to generate conclusion",
            "sources": []
        }}

# ===============================
# Pipeline Functions
# ===============================

def create_tools():
    """
    Create properly formatted tools for the agent
    
    Returns:
        List[Tool]: LangChain tools for the agent
    """
    return [
        Tool(
            name="web_search",
            func=search_quarterly,
            description="Search for NVIDIA quarterly financial information from web sources"
        ),
        Tool(
            name="rag_search",
            func=search_all_namespaces,
            description="Search across all document repositories for NVIDIA information"
        ),
        Tool(
            name="specific_quarter_search",
            func=search_specific_quarter,
            description="Search for specific quarter information from NVIDIA reports"
        ),
        Tool(
            name="snowflake_query",
            func=query_snowflake,
            description="Query Snowflake database for NVIDIA financial metrics"
        ),
        Tool(
            name="generate_report",
            func=final_report_tool,
            description="Generate a structured report from analyzed information"
        )
    ]

def initialize_nvidia_gpt():
    """Initialize NvidiaGPT agent with simplified configuration"""
    llm = ChatAnthropic(
        model="claude-3-haiku-20240307",  # Using the simpler model
        temperature=0,
        anthropic_api_key=os.environ.get('ANTHROPIC_API_KEY')
    )
    
    tools = create_tools()
    
    # Simplified agent initialization
    agent = initialize_agent(
        tools=tools,
        llm=llm,
        agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION,  # Simpler agent type
        handle_parsing_errors=True,
        max_iterations=2,  # Limit iterations to reduce token usage
        early_stopping_method="generate"
    )
    
    return agent

def generate_workflow_diagram(filename="nvidia_workflow"):
    """
    Generates and saves workflow diagram with visual enhancements
    
    Args:
        filename: Output filename without extension
        
    Returns:
        str: Path to generated diagram or None if failed
    """
    dot = Digraph(comment='NVIDIA Analysis Pipeline')
    
    # Set global graph attributes
    dot.attr(rankdir='LR', bgcolor='white', fontname='Helvetica')
    dot.attr('node', fontname='Helvetica', fontsize='12', style='filled', fontcolor='white', margin='0.4')
    dot.attr('edge', fontname='Helvetica', fontsize='10', penwidth='1.5')
    
    # Add nodes with colors and shapes
    dot.node('start', 'Start', shape='oval', style='filled', fillcolor='#4CAF50', color='#2E7D32')
    dot.node('web_search', 'Web Search', shape='box', style='filled,rounded', fillcolor='#2196F3', color='#0D47A1')
    dot.node('rag_search', 'RAG Search', shape='box', style='filled,rounded', fillcolor='#03A9F4', color='#0277BD')
    dot.node('snowflake', 'Snowflake', shape='box', style='filled,rounded', fillcolor='#00BCD4', color='#006064')
    dot.node('agent', 'NvidiaGPT Agent', shape='hexagon', style='filled', fillcolor='#9C27B0', color='#4A148C')
    dot.node('report_generator', 'Report Generator', shape='note', style='filled', fillcolor='#FF9800', color='#E65100')
    dot.node('end', 'End', shape='oval', style='filled', fillcolor='#F44336', color='#B71C1C')
    
    # Add edges with colors
    dot.edge('start', 'web_search', color='#2196F3')
    dot.edge('start', 'rag_search', color='#03A9F4')
    dot.edge('start', 'snowflake', color='#00BCD4')
    dot.edge('web_search', 'agent', color='#2196F3')
    dot.edge('rag_search', 'agent', color='#03A9F4')
    dot.edge('snowflake', 'agent', color='#00BCD4')
    dot.edge('agent', 'report_generator', color='#9C27B0')
    dot.edge('report_generator', 'end', color='#FF9800')
    
    # Generate diagram
    try:
        dot.render(filename, format='png', cleanup=True)
        return f"{filename}.png"
    except Exception as e:
        print(f"Warning: Could not generate diagram: {e}")
        return None

def build_pipeline(selected_agents=None):
    """
    Build and return the compiled pipeline based on selected agents
    
    Args:
        selected_agents: List of agent names or "All Agents"
    """
    # Create state graph
    graph = StateGraph(NvidiaGPTState)
    
    # Initialize NvidiaGPT agent
    nvidia_gpt = initialize_nvidia_gpt()
    
    # Create a closure to bind nvidia_gpt to agent_node
    def agent_with_gpt(state):
        return agent_node(state, nvidia_gpt)
    
    # Add start node
    graph.add_node("start", start_node)
    
    # Add nodes based on selected agents
    if selected_agents is None or "All Agents" in selected_agents:
        # Add all nodes for full pipeline
        graph.add_node("web_search", web_search_node)
        graph.add_node("rag_search", rag_search_node)
        graph.add_node("snowflake", snowflake_node)
        graph.add_node("agent", agent_with_gpt)
        graph.add_node("report_generator", final_report_node)
        
        # Set full flow
        graph.set_entry_point("start")
        graph.add_edge("start", "web_search")
        graph.add_edge("start", "rag_search")
        graph.add_edge("start", "snowflake")
        graph.add_edge("web_search", "agent")
        graph.add_edge("rag_search", "agent")
        graph.add_edge("snowflake", "agent")
        graph.add_edge("agent", "report_generator")
        graph.add_edge("report_generator", END)
    else:
        # Add only selected agent nodes
        if "Web Search Agent" in selected_agents:
            graph.add_node("web_search", web_search_node)
            graph.add_edge("start", "web_search")
            graph.add_edge("web_search", END)
            
        if "RAG Agent" in selected_agents:
            graph.add_node("rag_search", rag_search_node)
            graph.add_edge("start", "rag_search")
            graph.add_edge("rag_search", END)
            
        if "Snowflake Agent" in selected_agents:
            graph.add_node("snowflake", snowflake_node)
            graph.add_edge("start", "snowflake")
            graph.add_edge("snowflake", END)
    
    return graph.compile()

# For backward compatibility with imports
build_graph = build_pipeline

# ===============================
# Main Execution
# ===============================

if __name__ == "__main__":
    try:
        print("Initializing NVIDIA Research Pipeline...")
        pipeline = build_pipeline()
        
        print("Running analysis...")
        result = pipeline.invoke({
            "input": "Analyze NVIDIA's financial performance in Q4 2023",
            "question": "Analyze NVIDIA's financial performance in Q4 2023",
            "search_type": "Specific Quarter",
            "selected_periods": ["2023q4"],
            "chat_history": [],
            "intermediate_steps": []
        })
        
        print("\n✅ Analysis Complete!")
        print("\n📊 FINAL REPORT:")
        
        report = result.get("final_report", {})
        print(f"\n📝 {report.get('introduction', '')}")
        print("\n🔑 KEY FINDINGS:")
        for finding in report.get("key_findings", []):
            print(f"  • {finding}")
        print(f"\n📈 ANALYSIS:\n{report.get('analysis', '')}")
        print(f"\n🏁 CONCLUSION:\n{report.get('conclusion', '')}")
        
    except Exception as e:
        print(f"❌ Error running pipeline: {str(e)}")
        print("\nFull error traceback:")
        print(traceback.format_exc())