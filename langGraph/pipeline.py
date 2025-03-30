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
from graphviz import Digraph

# Add parent directory to path for agent imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import agent functions
from agents.websearch_agent import search_quarterly
from agents.rag_agent import search_all_namespaces, search_specific_quarter
from agents.snowflake_agent import query_snowflake, get_valuation_summary_with_llm_graph, get_ai_analysis_with_graph


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


def final_report_tool(input_dict: Dict) -> Dict:
    """Generates final report in structured format."""
    return {
        "introduction": input_dict.get("introduction", ""),
        "key_findings": input_dict.get("key_findings", []),
        "analysis": input_dict.get("analysis", ""),
        "conclusion": input_dict.get("conclusion", ""),
        "sources": input_dict.get("sources", [])
    }


def start_node(state: NvidiaGPTState) -> Dict:
    """Initial node that processes the input query."""
    return {"question": state["input"]}


def web_search_node(state: NvidiaGPTState) -> Dict:
    """Execute web search for NVIDIA information."""
    try:
        # Pass both the question and selected_periods from state to search_quarterly
        result = search_quarterly(
            query=state.get("question"),
            selected_periods=state.get("selected_periods")
        )
        return {"web_output": result}
    except Exception as e:
        return {"web_output": f"Web search error: {str(e)}"}

def rag_search_node(state: NvidiaGPTState) -> Dict:
    """Execute RAG search based on search type."""
    try:
        if state.get("search_type") == "All Quarters":
            # Search across all document namespaces
            result = search_all_namespaces(state["question"])
            return {"rag_output": {"type": "all", "result": result}}
        else:
            # For specific quarters
            input_dict = {
                "input_dict": {
                    "query": state["question"],
                    "selected_periods": state.get("selected_periods", ["2023q1"])
                }
            }
            result = search_specific_quarter.invoke(input_dict)
            return {
                "rag_output": {
                    "type": "specific",
                    "result": result,
                    "periods": state.get("selected_periods", ["2023q1"])
                }
            }
    except Exception as e:
        return {"rag_output": {"type": "error", "result": f"RAG search error: {str(e)}"}}


def snowflake_node(state: NvidiaGPTState) -> Dict:
    """Query Snowflake database for NVIDIA financial data with visualization."""
    try:
        # Get the user's original query
        query = state.get("question", "Analyze NVIDIA financial metrics")
        
        # Generate LLM-guided graph and analysis
        result = get_valuation_summary_with_llm_graph()
        
        if result["status"] == "failed":
            return {"snowflake_output": {"error": result["error"], "status": "failed"}}
        
        # Get AI analysis of the data with the generated graph
        analysis = get_ai_analysis_with_graph(query)
        
        # Return structured output with graph path and analysis
        return {
            "snowflake_output": {
                "data": result["summary"],
                "graph_path": result["chart_path"],
                "graph_specs": result["graph_specs"],
                "analysis": analysis,
                "status": "success"
            },
            "valuation_data": {
                "chart_path": result["chart_path"],
                "data": result["summary"]
            }
        }
    except Exception as e:
        print(f"Error in snowflake_node: {str(e)}")
        return {"snowflake_output": {"error": str(e), "status": "failed"}}
    

def agent_node(state: NvidiaGPTState, nvidia_gpt):
    """Execute NvidiaGPT agent with LLM, synthesizing data from all sources."""
    try:
        # Get user's original query
        user_query = state.get("question", "")
        
        # Get data from all sources
        raw_web = state.get("web_output", "")
        raw_rag = state.get("rag_output", {}).get("result", "")
        snowflake_data = state.get("snowflake_output", {})
        
        # Extract financial metrics from Snowflake
        financial_metrics = ""
        if isinstance(snowflake_data, dict) and snowflake_data.get("status") == "success":
            financial_metrics = snowflake_data.get("analysis", "")
            # Include path to chart if available
            chart_path = state.get("valuation_data", {}).get("chart_path", "")
            if chart_path:
                financial_metrics += f"\n\nFinancial chart available at: {chart_path}"

        # 1) Summarize Web Data (with focus on extracting image information)
        web_summary = "No web data available."
        web_images = []
        
        if raw_web:
            # Extract image URLs from web search results
            image_lines = []
            image_section = False
            for line in raw_web.split("\n"):
                if "Image Search Results" in line:
                    image_section = True
                    continue
                if image_section and "Thumbnail:" in line:
                    img_url = line.split("Thumbnail:")[1].strip()
                    web_images.append(img_url)
                    image_lines.append(f"- Image: {img_url}")
            
            prompt_for_web = f"""
You are analyzing NVIDIA's performance for: "{user_query}"

Below is web content from the latest news and search results:

{raw_web}

Please provide a concise summary (3-5 sentences) focusing on:
1. Direct answers to the user's specific question
2. Key financial metrics and trends mentioned
3. Recent news that impacts the company's performance

Format your response as a clear summary without using bullet points.
"""
            web_summary_result = nvidia_gpt.invoke({"input": prompt_for_web})
            if isinstance(web_summary_result, dict) and "output" in web_summary_result:
                web_summary = web_summary_result["output"]
            else:
                web_summary = str(web_summary_result)

        # 2) Summarize RAG Data with focus on key financial metrics
        rag_summary = "No historical data available."
        
        if raw_rag:
            prompt_for_rag = f"""
You are analyzing NVIDIA's historical performance for: "{user_query}"

Below is retrieved content from NVIDIA's official quarterly reports:

{raw_rag}

Please provide a concise summary focusing specifically on:
1. Key financial metrics directly related to the user's query
2. Revenue, profit margins, and growth rates by segment
3. Year-over-year comparisons that help answer the user's question

Focus on quantitative data and limit to 5-7 sentences.
"""
            rag_summary_result = nvidia_gpt.invoke({"input": prompt_for_rag})
            if isinstance(rag_summary_result, dict) and "output" in rag_summary_result:
                rag_summary = rag_summary_result["output"]
            else:
                rag_summary = str(rag_summary_result)

        # 3) Combine Everything for Final Analysis
        combined_prompt = f"""
USER QUERY: {user_query}

WEB SEARCH SUMMARY:
{web_summary}

HISTORICAL REPORTS SUMMARY:
{rag_summary}

FINANCIAL ANALYSIS:
{financial_metrics}

AVAILABLE VISUALIZATIONS: {len(web_images)} images found in search results.

Based on ALL the information above, please provide a comprehensive analysis of NVIDIA's performance 
that directly addresses the user's query: "{user_query}"

Your response must:
1. Begin with a direct answer to "{user_query}"
2. Include specific numbers and percentages from the data
3. Compare relevant performance metrics across periods
4. Highlight key strengths and any challenges
5. Provide business context that explains the underlying trends

Format as a clear, professional analysis with 2-3 paragraphs maximum.
"""
        final_response = nvidia_gpt.invoke({"input": combined_prompt})

        # Return the final answer plus individual summaries and image URLs for UI display
        if isinstance(final_response, dict) and "output" in final_response:
            return {
                "assistant_response": final_response["output"],
                "web_summary": web_summary,
                "rag_summary": rag_summary,
                "web_images": web_images  # Add images for use in final report
            }
        else:
            return {
                "assistant_response": str(final_response),
                "web_summary": web_summary,
                "rag_summary": rag_summary,
                "web_images": web_images
            }

    except Exception as e:
        return {
            "assistant_response": f"Analysis error: {str(e)}",
            "web_summary": "Error processing web data.",
            "rag_summary": "Error processing historical data.",
            "web_images": []
        }


def final_report_node(state: NvidiaGPTState) -> Dict:
    """Generate final report combining all sources in a clean, structured format."""
    try:
        # Get the original question and LLM-generated summary/analysis
        question = state.get("question", "NVIDIA performance analysis")
        assistant_response = state.get("assistant_response", "")
        
        # Get agent summaries and images
        rag_summary = state.get("rag_summary", "")
        web_summary = state.get("web_summary", "")
        web_images = state.get("web_images", [])
        
        # Initialize report sections
        introduction = f"# NVIDIA Analysis: {question}\n\n"
        executive_summary = "## Executive Summary\n\n"
        web_insights = "## Current Market Insights\n\n"
        visualizations = "## Visualizations\n\n"
        historical_data = "## Historical Performance\n\n"
        financial_metrics = "## Financial Metrics\n\n"
        analysis_section = "## Expert Analysis\n\n"
        conclusion = "## Conclusion\n\n"
        sources = "## Sources\n\n"
        
        # Build Executive Summary section
        if assistant_response:
            executive_summary += assistant_response.strip() + "\n\n"
        else:
            executive_summary += "Analysis based on available data from financial reports, web sources, and metrics.\n\n"
        
        # Process Web Search results
        raw_web = state.get("web_output", "")
        if web_summary and "No web data" not in web_summary:
            web_insights += f"{web_summary}\n\n"
        elif raw_web:
            # Extract just the relevant web search insights
            # Find the top highlights from the raw web search
            web_lines = raw_web.split("\n")
            top_results = []
            for i, line in enumerate(web_lines):
                if line.strip().startswith(("1.", "2.", "3.")) and i+1 < len(web_lines):
                    top_results.append(f"- {web_lines[i+1].strip()}")
            
            if top_results:
                web_insights += "### Latest Findings\n\n"
                web_insights += "\n".join(top_results) + "\n\n"
        
        # Add visualizations section with embedded images
        if web_images:
            visualizations += "### NVIDIA Performance Visualizations\n\n"
            for i, img_url in enumerate(web_images[:3]):  # Limit to first 3 images
                visualizations += f"#### Visualization {i+1}\n\n"
                visualizations += f"![NVIDIA Visualization {i+1}]({img_url})\n\n"
        else:
            visualizations += "No visualizations available for this query.\n\n"
        
        # Process RAG results
        raw_rag_data = state.get("rag_output", {})
        raw_rag_text = raw_rag_data.get("result", "")
        
        if rag_summary and "No RAG data" not in rag_summary:
            historical_data += f"{rag_summary}\n\n"
            
            # Try to extract a financial table if present
            table_data = extract_financial_table(raw_rag_text)
            if table_data:
                historical_data += "\n### Key Financial Data\n\n"
                historical_data += table_data + "\n\n"
        elif raw_rag_text:
            # Pull key financial metrics from RAG results when available
            historical_data += "### Historical Performance Highlights\n\n"
            
            # Extract revenue and growth information from RAG content
            revenue_data = []
            
            if "Revenue" in raw_rag_text and "$" in raw_rag_text:
                # Find lines with revenue numbers
                rag_lines = raw_rag_text.split("\n")
                for line in rag_lines:
                    if "Revenue" in line and "$" in line:
                        clean_line = line.replace("|", "").strip()
                        revenue_data.append(f"- {clean_line}")
            
            if revenue_data:
                historical_data += "#### Revenue Performance\n\n"
                historical_data += "\n".join(revenue_data[:3]) + "\n\n"
            else:
                historical_data += "Key financial data available in the quarterly reports. See analysis for insights.\n\n"
        
        # Process Snowflake metrics and chart
        snowflake_output = state.get('snowflake_output', {})
        valuation_data = state.get('valuation_data', {})
        
        if isinstance(snowflake_output, dict) and snowflake_output.get("status") == "success":
            financial_metrics += "### Latest Financial Metrics\n\n"
            
            # Reference to chart if available
            chart_path = valuation_data.get('chart_path', '')
            if chart_path:
                financial_metrics += f"![NVIDIA Financial Metrics]({chart_path})\n\n"
            
            # Format financial data in a clean table if available
            financial_data = format_financial_data(valuation_data.get('data', {}))
            if financial_data:
                financial_metrics += "### Key Financial Indicators\n\n"
                financial_metrics += financial_data + "\n\n"
            
            # Add analysis from the Snowflake agent
            analysis = snowflake_output.get('analysis', '')
            if analysis and isinstance(analysis, str):
                financial_metrics += "### Financial Analysis\n\n"
                financial_metrics += analysis.strip() + "\n\n"
        
        # Build the analysis section using the LLM's response
        if assistant_response:
            # Extract key insights as bullet points
            analysis_section += "Based on the collected data, NVIDIA's performance shows the following key patterns:\n\n"
            
            # Extract 3-5 bullet points of insights
            analysis_points = []
            for sentence in assistant_response.split(". "):
                if len(sentence) > 20 and not any(sentence.strip() in p for p in analysis_points):
                    analysis_points.append(f"- {sentence.strip()}.")
                if len(analysis_points) >= 4:
                    break
            
            analysis_section += "\n".join(analysis_points) + "\n\n"
        
        # Conclusion section - create a concise wrap-up
        conclusion += "NVIDIA continues to demonstrate leadership in the GPU and AI computing markets. "
        
        if assistant_response and ("down" in assistant_response.lower() or "decline" in assistant_response.lower()):
            conclusion += "While facing some market challenges and inventory adjustments, "
        
        if assistant_response and ("growth" in assistant_response.lower() or "increase" in assistant_response.lower()):
            conclusion += "With strong growth in key segments, particularly Data Center, "
            
        conclusion += "the company is well-positioned for future opportunities in AI, data center acceleration, "
        conclusion += "and next-generation computing platforms.\n\n"
        
        # Build sources list
        sources_list = []
        if "Web Search Agent" in state.get("selected_agents", []):
            sources_list.append("- **Web Search**: Latest news and market data")
        
        if "RAG Agent" in state.get("selected_agents", []):
            rag_type = raw_rag_data.get("type", "all")
            periods = raw_rag_data.get("periods", [""])
            periods_str = ", ".join(periods) if isinstance(periods, list) else str(periods)
            if rag_type == "specific" and periods_str:
                sources_list.append(f"- **NVIDIA Quarterly Reports**: {periods_str}")
            else:
                sources_list.append("- **NVIDIA Quarterly Reports**: Historical data")
        
        if "Snowflake Agent" in state.get("selected_agents", []):
            sources_list.append("- **Financial Database**: Valuation metrics and analysis")
            
        sources += "\n".join(sources_list) + "\n\n"
        
        # Combine all sections
        full_report = (
            introduction +
            executive_summary +
            web_insights +
            visualizations +
            historical_data +
            financial_metrics +
            analysis_section +
            conclusion +
            sources
        )
        
        # Return the structured report
        return {
            "final_report": {
                "formatted_report": full_report,
                "introduction": introduction.strip(),
                "executive_summary": executive_summary.strip(),
                "web_insights": web_insights.strip(),
                "visualizations": visualizations.strip(),
                "historical_data": historical_data.strip(),
                "financial_metrics": financial_metrics.strip(),
                "analysis": analysis_section.strip(),
                "conclusion": conclusion.strip(),
                "sources": sources.strip()
            }
        }

    except Exception as e:
        print(f"Error in final_report_node: {e}")
        return {
            "final_report": {
                "formatted_report": f"# Error Generating Report\n\nAn error occurred: {str(e)}",
                "introduction": "Error generating report",
                "key_findings": [f"Error: {str(e)}"],
                "analysis": "Analysis unavailable due to error",
                "conclusion": "Unable to generate conclusion",
                "sources": []
            }
        }
# Helper functions for report formatting
def extract_financial_table(raw_text):
    """Extract financial tables from RAG results and format as markdown."""
    if "|" not in raw_text:
        return ""
    
    table_lines = []
    in_table = False
    
    for line in raw_text.split("\n"):
        if "|" in line and "---" in line:
            in_table = True
            table_lines.append(line)
        elif in_table and "|" in line:
            table_lines.append(line)
        elif in_table and "|" not in line:
            in_table = False
            table_lines.append("\n")
    
    if table_lines:
        return "\n".join(table_lines)
    return ""

def format_financial_data(data):
    """Format financial data as a clean markdown table."""
    if not data or not isinstance(data, dict):
        return ""
    
    table = "| Metric | Value |\n| ------ | ----- |\n"
    
    for key, value in data.items():
        if isinstance(value, (int, float)):
            formatted_value = f"${value:,.2f}" if value > 1 else f"{value:.3f}"
        else:
            formatted_value = str(value)
        
        table += f"| {key} | {formatted_value} |\n"
    
    return table

def create_tools():
    """Create LangChain tools for the agent."""
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
    """Initialize NvidiaGPT agent with simplified configuration."""
    llm = ChatAnthropic(
        model="claude-3-haiku-20240307",
        temperature=0,
        anthropic_api_key=os.environ.get('ANTHROPIC_API_KEY')
    )
    tools = create_tools()
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

    agent = initialize_agent(
        tools=tools,
        llm=llm,
        agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
        memory=memory,
        handle_parsing_errors=True,
        max_iterations=3,
        verbose=False
    )
    return agent


def generate_workflow_diagram(filename="nvidia_workflow"):
    """Generates and saves workflow diagram with visual enhancements."""
    dot = Digraph(comment='NVIDIA Analysis Pipeline')
    dot.attr(rankdir='LR', bgcolor='white', fontname='Helvetica')
    dot.attr('node', fontname='Helvetica', fontsize='12', style='filled', fontcolor='white', margin='0.4')
    dot.attr('edge', fontname='Helvetica', fontsize='10', penwidth='1.5')

    # Nodes
    dot.node('start', 'Start', shape='oval', style='filled', fillcolor='#4CAF50', color='#2E7D32')
    dot.node('web_search', 'Web Search', shape='box', style='filled,rounded', fillcolor='#2196F3', color='#0D47A1')
    dot.node('rag_search', 'RAG Search', shape='box', style='filled,rounded', fillcolor='#03A9F4', color='#0277BD')
    dot.node('snowflake', 'Snowflake', shape='box', style='filled,rounded', fillcolor='#00BCD4', color='#006064')
    dot.node('agent', 'NvidiaGPT Agent', shape='hexagon', style='filled', fillcolor='#9C27B0', color='#4A148C')
    dot.node('report_generator', 'Report Generator', shape='note', style='filled', fillcolor='#FF9800', color='#E65100')
    dot.node('end', 'End', shape='oval', style='filled', fillcolor='#F44336', color='#B71C1C')

    # Edges
    dot.edge('start', 'web_search', color='#2196F3')
    dot.edge('start', 'rag_search', color='#03A9F4')
    dot.edge('start', 'snowflake', color='#00BCD4')
    dot.edge('web_search', 'agent', color='#2196F3')
    dot.edge('rag_search', 'agent', color='#03A9F4')
    dot.edge('snowflake', 'agent', color='#00BCD4')
    dot.edge('agent', 'report_generator', color='#9C27B0')
    dot.edge('report_generator', 'end', color='#FF9800')

    try:
        dot.render(filename, format='png', cleanup=True)
        return f"{filename}.png"
    except Exception as e:
        print(f"Warning: Could not generate diagram: {e}")
        return None


def build_pipeline(selected_agents: List[str] = None):
    """
    Build and return the compiled pipeline with dynamic agent selection.
    The flow ensures all selected agents connect to the agent node first
    before report generation, optimized for different agent combinations.
    """
    if selected_agents is None:
        selected_agents = []

    graph = StateGraph(NvidiaGPTState)

    # Make "start" a real node
    graph.add_node("start", start_node)
    graph.set_entry_point("start")

    # Add the agents in optimal order based on data dependencies
    last_node = "start"
    
    # Optimal processing order: RAG -> Web Search -> Snowflake
    # This provides historical context before current data before financial metrics
    
    # If RAG is selected, add it first (provides historical context)
    if "RAG Agent" in selected_agents:
        graph.add_node("rag_search", rag_search_node)
        graph.add_edge(last_node, "rag_search")
        last_node = "rag_search"
    
    # Web Search can benefit from RAG context if available
    if "Web Search Agent" in selected_agents:
        graph.add_node("web_search", web_search_node)
        graph.add_edge(last_node, "web_search")
        last_node = "web_search"
    
    # Snowflake analysis works best with context from both RAG and web search
    if "Snowflake Agent" in selected_agents:
        graph.add_node("snowflake", snowflake_node)
        graph.add_edge(last_node, "snowflake")
        last_node = "snowflake"

    # Add the agent node to process all collected information
    nvidia_gpt = initialize_nvidia_gpt()
    graph.add_node("agent", lambda state: agent_node(state, nvidia_gpt))
    
    # Connect the last data-gathering node to the agent
    graph.add_edge(last_node, "agent")
    
    # Add report generator node
    graph.add_node("report_generator", final_report_node)
    
    # Agent always connects to report generator
    graph.add_edge("agent", "report_generator")
    
    # Finally connect to END
    graph.add_edge("report_generator", END)

    return graph.compile()

if __name__ == "__main__":
    try:
        # Example with all agents for comprehensive analysis
        pipeline = build_pipeline(["RAG Agent", "Web Search Agent", "Snowflake Agent"])

        # Define a clear test query
        test_query = "Analyze NVIDIA's financial performance in Q4 2023"
        
        # Run pipeline with detailed state
        result = pipeline.invoke({
            "input": test_query,
            "question": test_query,
            "search_type": "All Quarters",
            # "selected_periods": ["2023q4"],
            "selected_agents": ["RAG Agent", "Web Search Agent", "Snowflake Agent"],
            "chat_history": [],
            "intermediate_steps": []
        })

        print("\n✅ Analysis Complete!")
        
        # Extract the formatted report for display
        final_report = result.get("final_report", {})
        formatted_report = final_report.get("formatted_report", "No report generated")
        
        # Print the complete formatted report for Streamlit markdown display
        print(formatted_report)

    except Exception as e:
        print(f"❌ Error running pipeline: {str(e)}")
        print("\nFull error traceback:")
        print(traceback.format_exc())