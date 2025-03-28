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
from agents.snowflake_agent import get_valuation_summary_with_llm_graph, get_ai_analysis_with_graph


class NvidiaGPTState(TypedDict, total=False):
    """State definition for NVIDIA research pipeline"""
    input: str  # User's original query
    question: str  # Processed question
    search_type: str  # "All Quarters" or "Specific Quarter"
    selected_periods: List[str]  # List of quarters to analyze
    web_output: str  # Results from web search
    web_links: List[str]  # Links from web search results
    web_images: List[str]  # Image URLs from web search results
    rag_output: Dict[str, Any]  # Results from RAG search
    snowflake_output: Dict[str, Any]  # Results from Snowflake query
    valuation_data: Dict[str, Any]  # Financial visualization data
    chat_history: List[Dict[str, Any]]  # Conversation history
    intermediate_steps: Annotated[List[tuple[AgentAction, str]], operator.add]  # Agent reasoning steps
    assistant_response: str  # Agent's response
    final_report: Dict[str, Any]  # Final structured report
    model: str  # Model name for LLM



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
        if isinstance(result, dict):

            return {"web_output": result.get("text", ""),
                    "web_links": result.get("links", []),
                    "web_images": result.get("images", [])
                    }
    except Exception as e:
        return {"web_output": f"Web search error: {str(e)}"
                }

def rag_search_node(state: NvidiaGPTState) -> Dict:
    """Execute RAG search based on search type."""
    try:
        # Extract model name from state
        model_name = state.get("model")
        
        if state.get("search_type") == "All Quarters":
            # Search across all document namespaces with model_name
            result = search_all_namespaces(state["question"], model_name=model_name)
            return {"rag_output": {"type": "all", "result": result}}
        else:
            # For specific quarters
            input_dict = {
                "input_dict": {
                    "query": state["question"],
                    "selected_periods": state.get("selected_periods", ["2023q1"]),
                    "model_name": model_name
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
        query = state.get("question")
        model_name = state.get("model")
        
        print(f"Running Snowflake analysis with model: {model_name}")
        
        # Get AI analysis which includes valuation summary and visualization
        result = get_ai_analysis_with_graph(query, model_name=model_name)
        
        # Handle result as dictionary or string
        if isinstance(result, dict) and result.get("status") == "success":
            # Ensure chart_path exists and is accessible
            chart_path = result.get("chart_path", "")
            
            # Try to use default location if path is missing
            if not chart_path or not os.path.exists(chart_path):
                print(f"Warning: Chart path not found: {chart_path}")
                if os.path.exists('llm_generated_graph.png'):
                    chart_path = 'llm_generated_graph.png'
                    result['chart_path'] = chart_path
            
            # Return full result object with separate chart path for easy access
            return {
                "snowflake_output": result,
                "valuation_data": {
                    "chart_path": chart_path,
                    "data": result.get("summary", {})
                }
            }
        elif isinstance(result, str):
            # If result is a string, it's likely an error message
            print(f"Snowflake agent returned string: {result}")
            return {
                "snowflake_output": {
                    "error": result,
                    "status": "failed",
                    "analysis": "Using RAG agent's financial analysis directly",  # Fallback
                    "chart_path": "llm_generated_graph.png" if os.path.exists('llm_generated_graph.png') else ""
                }
            }
        else:
            # Handle other unexpected cases
            print(f"Snowflake agent returned unexpected format: {type(result)}")
            return {
                "snowflake_output": {
                    "error": "Invalid response format from Snowflake agent",
                    "status": "failed",
                    "analysis": "Using RAG agent's financial analysis directly",  # Fallback
                    "chart_path": "llm_generated_graph.png" if os.path.exists('llm_generated_graph.png') else ""
                }
            }
    except Exception as e:
        print(f"Error in snowflake_node: {str(e)}")
        return {
            "snowflake_output": {
                "error": str(e), 
                "status": "failed",
                "analysis": "Using RAG agent's financial analysis directly",  # Fallback
                "chart_path": "llm_generated_graph.png" if os.path.exists('llm_generated_graph.png') else ""
            }
        }
    
def agent_node(state: NvidiaGPTState, nvidia_gpt):
    """Execute NvidiaGPT agent with LLM, synthesizing data from all sources."""
    try:
        # Get user's original query
        user_query = state.get("question", "")
        
        # Get data from all sources
        raw_web = state.get("web_output", "")
        web_images = state.get("web_images", [])
        web_links = state.get("web_links", [])
        
        # Get RAG data
        raw_rag_data = state.get("rag_output", {})
        rag_insights = ""
        
        if isinstance(raw_rag_data, dict):
            result = raw_rag_data.get("result", {})
            if isinstance(result, dict):
                rag_insights = result.get("insights", "")
        
        # Get Snowflake data
        snowflake_data = state.get("snowflake_output", {})
        financial_analysis = ""
        
        if isinstance(snowflake_data, dict):
            financial_analysis = snowflake_data.get("analysis", "")
        
        # Set defaults if no data available
        web_summary = "No web data available."
        if raw_web:
            if isinstance(raw_web, dict):
                raw_web = raw_web.get("text", "")
            
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
            web_summary_result = nvidia_gpt.invoke(prompt_for_web)
            web_summary = web_summary_result.content if hasattr(web_summary_result, 'content') else str(web_summary_result)

        # Get RAG summary
        rag_summary = "No historical data available."
        if rag_insights:
            rag_summary = rag_insights
        
        # Generate final comprehensive analysis
        combined_prompt = f"""
USER QUERY: {user_query}

WEB SEARCH SUMMARY:
{web_summary}

HISTORICAL REPORTS SUMMARY:
{rag_summary}

FINANCIAL ANALYSIS:
{financial_analysis}

AVAILABLE VISUALIZATIONS: {len(web_images)} images found in search results.

Based on ALL the information above, provide a comprehensive analysis of NVIDIA's performance 
that directly addresses the user's query: "{user_query}"

Your response must:
1. Begin with a direct answer to "{user_query}"
2. Include specific numbers and percentages from the data
3. Compare relevant performance metrics across periods
4. Highlight key strengths and any challenges
5. Provide business context that explains the underlying trends

Format as a clear, professional analysis with proper paragraph breaks. DO NOT use any special formatting that might break in markdown.
Avoid using non-standard spacing, character formatting, or any other elements that might cause display issues.
"""
        final_response = nvidia_gpt.invoke(combined_prompt)
        
        # Clean the response to fix formatting issues
        response_text = final_response.content if hasattr(final_response, 'content') else str(final_response)
        cleaned_response = response_text.replace('\n\n\n', '\n\n').strip()
        
        return {
            "assistant_response": cleaned_response,
            "web_summary": web_summary,
            "rag_summary": rag_summary,
            "web_images": web_images,
            "web_links": web_links
        }

    except Exception as e:
        print(f"Error in agent_node: {str(e)}")
        return {
            "assistant_response": f"Analysis error: {str(e)}",
            "web_summary": "Error processing web data.",
            "rag_summary": "Error processing historical data.",
            "web_images": [],
            "web_links": []
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
        web_links = state.get("web_links", [])
        
        # Get Snowflake data
        snowflake_output = state.get('snowflake_output', {})
        valuation_data = state.get('valuation_data', {})
        
        # Initialize report sections
        introduction = f"# NVIDIA Analysis: {question}\n\n"
        executive_summary = "## Executive Summary\n\n"
        web_insights = "## Current Market Insights\n\n"
        visualizations = "## Visualizations\n\n### NVIDIA Performance Visualizations\n\n"
        historical_data = "## Historical Performance\n\n### Historical Financial Performance\n\n"
        financial_metrics = "## Financial Metrics\n\n"
        analysis_section = "## Expert Analysis\n\n"
        conclusion = "## Conclusion\n\n"
        sources = "## Sources\n\n"
        
        # Process Web Search results to extract insights
        if web_summary and isinstance(web_summary, str) and "No web data" not in web_summary:
            web_insights += f"{web_summary}\n\n"
            
            # Add links if available for context
            if web_links and isinstance(web_links, list) and len(web_links) > 0:
                web_insights += "#### Key Sources:\n"
                for i, link in enumerate(web_links[:3]):
                    try:
                        domain = link.split('/')[2]
                        web_insights += f"- [{domain}]({link})\n"
                    except:
                        pass
                web_insights += "\n"
        else:
            web_insights = ""  # Skip this section entirely if no data
            
        # Process RAG results to extract insights
        if rag_summary and isinstance(rag_summary, str) and "No historical data" not in rag_summary:
            historical_data += f"{rag_summary}\n\n"
        else:
            historical_data = ""  # Skip this section entirely if no data
            
        # Process visualizations
        # Get chart path with fallbacks
        # Process visualizations
        # Get chart path with fallbacks
        chart_path = snowflake_output.get('chart_path', '')
        if not chart_path and isinstance(valuation_data, dict):
            chart_path = valuation_data.get('chart_path', '')
            
        has_visuals = False
        # Try to encode the image as base64 for embedding
        if chart_path and os.path.exists(chart_path):
            try:
                import base64
                with open(chart_path, "rb") as img_file:
                    b64_string = base64.b64encode(img_file.read()).decode()
                
                # Add embedded image in markdown
                financial_metrics += "#### Financial visualization:\n"
                financial_metrics += f"![NVIDIA Financial Metrics](data:image/png;base64,{b64_string})\n\n"
                has_financial_metrics = True
                
                # Also add direct file reference as fallback
                financial_metrics += f"*(Image path: {chart_path})*\n\n"
            except Exception as e:
                # Fallback to regular file reference
                print(f"Error encoding image: {str(e)}")
                financial_metrics += "#### Financial visualization:\n"
                financial_metrics += f"![NVIDIA Financial Metrics]({chart_path})\n\n"
                has_financial_metrics = True
        
        # Add web images if available
        if web_images and isinstance(web_images, list):
            for i, img_url in enumerate(web_images[:3]):
                if img_url and isinstance(img_url, str):
                    visualizations += f"#### Web Result {i+1}\n\n"
                    visualizations += f"![NVIDIA Visualization {i+1}]({img_url})\n\n"
                    has_visuals = True
                
        if not has_visuals:
            visualizations = ""  # Skip this section entirely if no visuals
            
        # Process Snowflake metrics and chart
        has_financial_metrics = False
        if isinstance(snowflake_output, dict):
            # Always add the financial metrics section
            financial_metrics += "### Latest Financial Metrics\n\n"
            
            # Add analysis from the Snowflake agent
            analysis = snowflake_output.get('analysis', '')
            if analysis and isinstance(analysis, str):
                financial_metrics += analysis.strip() + "\n\n"
                has_financial_metrics = True
            
            # Always check for chart path and add it to financial metrics section
            chart_path = snowflake_output.get('chart_path', '')
            if not chart_path and isinstance(valuation_data, dict):
                chart_path = valuation_data.get('chart_path', '')
                
            if chart_path and os.path.exists(chart_path):
                financial_metrics += "#### Financial visualization:\n"
                financial_metrics += f"![NVIDIA Financial Metrics]({chart_path})\n\n"
                has_financial_metrics = True

        # If no financial metrics could be added, provide a message
        if not has_financial_metrics:
            financial_metrics += "No detailed financial metrics were available for this query.\n\n"
            
        # Build the analysis section using the LLM's response
        if assistant_response and isinstance(assistant_response, str):
            # Clean up any formatting issues in the response
            clean_response = assistant_response.strip().replace("\n\n\n", "\n\n")
            
            # Extract key insights from the response
            analysis_section += f"{clean_response}\n\n"
        else:
            analysis_section = ""  # Skip this section if no analysis
            
        # Build executive summary
        if assistant_response and isinstance(assistant_response, str):
            # Create a well-formatted executive summary
            executive_summary += "### Key Findings\n\n"
            
            # Extract first paragraph as main summary
            paragraphs = assistant_response.split("\n\n")
            if paragraphs:
                executive_summary += f"{paragraphs[0].strip()}\n\n"
            
            # Add highlights section with bullet points
            executive_summary += "### Highlights\n\n"
            
            # Extract financial metrics from assistant response
            metrics_found = False
            for para in paragraphs[1:3]:  # Look in the next couple of paragraphs for metrics
                if any(term in para.lower() for term in ['revenue', 'growth', 'increase', 'profit', 'earnings', '$', '%']):
                    key_points = para.split('. ')
                    for point in key_points[:3]:  # Limit to first 3 points for conciseness
                        if point.strip():
                            executive_summary += f"- {point.strip()}.\n"
                            metrics_found = True
            
            # If no metrics were found, add some from web_summary
            if not metrics_found and web_summary and isinstance(web_summary, str):
                web_insights = web_summary.split('. ')
                for insight in web_insights[:2]:  # Limit to first 2 insights
                    if insight.strip():
                        executive_summary += f"- {insight.strip()}.\n"
            
            # Add market context from rag_summary or financial metrics
            executive_summary += "\n### Market Context\n\n"
            
            if financial_metrics and "no detailed financial metrics" not in financial_metrics.lower():
                # Extract key financial context
                sentences = [s.strip() + "." for s in financial_metrics.replace("\n", " ").split(".") if s.strip()]
                for sentence in sentences[:2]:  # First 2 sentences
                    if any(term in sentence.lower() for term in ["market", "industry", "position", "competitor", "trend"]):
                        executive_summary += f"{sentence} "
            
            # Add fallback if no context was found
            if "Market Context" in executive_summary and len(executive_summary.split("Market Context")[1].strip()) < 5:
                if isinstance(rag_summary, str) and rag_summary.strip():
                    executive_summary += "Based on historical data and current market conditions, "
                    executive_summary += "NVIDIA continues to demonstrate strong market positioning in the GPU and AI computing sectors.\n\n"
                else:
                    executive_summary += "NVIDIA's performance should be evaluated in the context of the broader semiconductor industry and AI market trends.\n\n"
        else:
            # Fallback if no assistant response is available
            executive_summary += "### Key Findings\n\n"
            executive_summary += "Analysis of NVIDIA's performance based on available financial data, market reports, and industry trends.\n\n"
            
            executive_summary += "### Highlights\n\n"
            executive_summary += "- Financial and operational metrics from multiple sources.\n"
            executive_summary += "- Recent developments affecting NVIDIA's market position.\n"
            executive_summary += "- Performance indicators across key business segments.\n\n"
            
            executive_summary += "### Market Context\n\n"
            executive_summary += "NVIDIA's results should be viewed within the context of industry trends, competitive dynamics, and overall technology sector performance.\n\n"
            
        # Build sources list
        sources_list = []
        if "Web Search Agent" in state.get("selected_agents", []):
            source_text = "- **Web Search**: Latest news and market data"
            if web_links:
                source_text += f" ({len(web_links)} sources)"
            sources_list.append(source_text)
        
        if "RAG Agent" in state.get("selected_agents", []):
            source_text = "- **NVIDIA Quarterly Reports**: Historical financial data"
            if state.get("selected_periods") and state["selected_periods"] != ["all"]:
                source_text += f" (Periods: {', '.join(state['selected_periods'])})"
            sources_list.append(source_text)
        
        if "Snowflake Agent" in state.get("selected_agents", []):
            sources_list.append("- **Financial Database**: Valuation metrics and technical analysis")
            
        sources += "\n".join(sources_list) + "\n\n"
        
        # Build conclusion
        if isinstance(assistant_response, str):
            # Extract conclusion from last paragraph if possible
            paragraphs = assistant_response.split("\n\n")
            if paragraphs and len(paragraphs) > 1:
                conclusion += paragraphs[-1] + "\n\n"
            else:
                conclusion += "NVIDIA continues to demonstrate leadership in the GPU and AI computing markets. "
                if "growth" in assistant_response.lower() or "increase" in assistant_response.lower():
                    conclusion += "With strong growth in key segments, particularly Data Center, "
                conclusion += "the company is well-positioned for future opportunities in AI and next-generation computing.\n\n"
        else:
            conclusion += "NVIDIA continues to demonstrate leadership in the GPU and AI computing markets. "
            conclusion += "The company is well-positioned for future opportunities in AI and next-generation computing.\n\n"
        
        # Combine sections, but only include non-empty ones
        sections = []
        if introduction.strip():
            sections.append(introduction)
        if executive_summary.strip():
            sections.append(executive_summary)
        if web_insights.strip():
            sections.append(web_insights)
        if visualizations.strip():
            sections.append(visualizations)
        if historical_data.strip():
            sections.append(historical_data)
        if financial_metrics.strip():
            sections.append(financial_metrics)
        if analysis_section.strip():
            sections.append(analysis_section)
        if conclusion.strip():
            sections.append(conclusion)
        if sources.strip():
            sections.append(sources)
            
        full_report = "\n".join(sections)
        
        # Return the structured report
        return {
            "final_report": full_report,  # Single string for API compatibility
            "formatted_report": full_report,
            "introduction": introduction.strip(),
            "executive_summary": executive_summary.strip(),
            "web_insights": web_insights.strip() or "No current market data available for this query.",
            "visualizations": visualizations.strip() or "No visualizations available for this query.",
            "historical_data": historical_data.strip() or "No historical financial data available for this query.",
            "financial_metrics": financial_metrics.strip() or "No financial metrics available for this query.",
            "analysis": analysis_section.strip() or "No expert analysis available for this query.",
            "conclusion": conclusion.strip(),
            "sources": sources.strip(),
            "web_images": web_images if isinstance(web_images, list) else [],
            "web_links": web_links if isinstance(web_links, list) else []
        }
    except Exception as e:
        print(f"Error in final_report_node: {str(e)}")
        import traceback
        traceback.print_exc()
        
        # Return a basic error report
        return {
            "final_report": f"# Error Generating Report\n\nAn error occurred: {str(e)}",
            "formatted_report": f"# Error Generating Report\n\nAn error occurred: {str(e)}",
            "introduction": "Error generating report",
            "executive_summary": f"Error: {str(e)}",
            "web_insights": "Error retrieving market insights.",
            "visualizations": "Error loading visualizations.",
            "historical_data": "Error retrieving historical data.",
            "financial_metrics": "Error retrieving financial metrics.",
            "analysis": "Analysis unavailable due to error.",
            "conclusion": "Unable to generate conclusion due to error.",
            "sources": "Sources unavailable due to error."
        }


def initialize_nvidia_gpt(model_name="claude-3-haiku-20240307"):
    """Initialize NvidiaGPT agent with the selected model."""
    try:
        # Map model prefixes to appropriate LangChain classes
        model_map = {
            "claude": ("langchain_anthropic", "ChatAnthropic", "ANTHROPIC_API_KEY"),
            "gemini": ("langchain_google_genai", "ChatGoogleGenerativeAI", "GEMINI_API_KEY"),
            "deepseek": ("langchain_openai", "ChatOpenAI", "DEEP_SEEK_API_KEY"),  # Update to match snowflake_agent.py
            "grok": ("langchain_groq", "ChatGroq", "GROK_API_KEY")  # Update to match snowflake_agent.py
        }
        
        # Find the matching model provider
        for prefix, (module_name, class_name, api_key_name) in model_map.items():
            if prefix in model_name.lower():
                # Dynamically import the module and class
                module = __import__(module_name, fromlist=[class_name])
                model_class = getattr(module, class_name)
                
                # Create and return the LLM instance
                return model_class(
                    model=model_name,
                    temperature=0,
                    api_key=os.environ.get(api_key_name)
                )
        
        # Default to Claude Haiku if no matching provider
        return ChatAnthropic(
            model="claude-3-haiku-20240307",
            temperature=0,
            anthropic_api_key=os.environ.get('ANTHROPIC_API_KEY')
        )
    except Exception as e:
        print(f"Error initializing LLM: {str(e)}. Falling back to default Claude model.")
        return ChatAnthropic(
            model="claude-3-haiku-20240307",
            temperature=0,
            anthropic_api_key=os.environ.get('ANTHROPIC_API_KEY')
        )
    
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


def build_pipeline(selected_agents: List[str] = None, model: str = "claude-3-haiku-20240307"):
    """Build and return the compiled pipeline with dynamic agent selection."""
    if not selected_agents:
        selected_agents = []  # Use empty list instead of None

    graph = StateGraph(NvidiaGPTState)
    
    # Add nodes and connect them in optimal order
    nodes_added = {}
    
    # Add start node
    graph.add_node("start", start_node)
    graph.set_entry_point("start")
    last_node = "start"
    
    # Add agent nodes in optimal order: RAG -> Web -> Snowflake
    agent_nodes = {
        "RAG Agent": ("rag_search", rag_search_node),
        "Web Search Agent": ("web_search", web_search_node),
        "Snowflake Agent": ("snowflake", snowflake_node)
    }
    
    for agent, (node_name, node_func) in agent_nodes.items():
        if agent in selected_agents:
            graph.add_node(node_name, node_func)
            graph.add_edge(last_node, node_name)
            last_node = node_name
            nodes_added[agent] = node_name
    
    # Initialize LLM
    nvidia_gpt = initialize_nvidia_gpt(model)
    
    # Add synthesis and report nodes
    graph.add_node("agent", lambda state: agent_node(state, nvidia_gpt))
    graph.add_node("report_generator", final_report_node)
    
    # Connect last data node to agent
    graph.add_edge(last_node, "agent")
    graph.add_edge("agent", "report_generator")
    graph.add_edge("report_generator", END)
    
    return graph.compile()

if __name__ == "__main__":
    try:
        # Example with all agents for comprehensive analysis
        pipeline = build_pipeline(["RAG Agent", "Web Search Agent", "Snowflake Agent"],model="gemini-1.5-flash")

        # Define a clear test query
        test_query = "Analyze NVIDIA's financial performance in Q4 2023"
        
        # Run pipeline with detailed state
        result = pipeline.invoke({
            "input": test_query,
            "question": test_query,
            "search_type": "Specific Quarter",
            "selected_periods": ["2023q4"],
            "selected_agents": ["RAG Agent", "Web Search Agent", "Snowflake Agent"],
            "chat_history": [],
            "intermediate_steps": []
        })

        print("\n✅ Analysis Complete!")
        
        # Fix: Check the type of final_report and handle appropriately
        final_report = result.get("final_report")
        
        # If final_report is a dictionary, get formatted_report from it
        # Otherwise, assume it's already the formatted content
        if isinstance(final_report, dict):
            formatted_report = final_report.get("formatted_report", "No report generated")
        else:
            formatted_report = final_report if final_report else "No report generated"
        
        # Print the complete formatted report for Streamlit markdown display
        print(formatted_report)

    except Exception as e:
        print(f"❌ Error running pipeline: {str(e)}")
        print("\nFull error traceback:")
        print(traceback.format_exc())