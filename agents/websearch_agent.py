# agents/websearch_agent.py
import os
from typing import Dict, Any
from serpapi import GoogleSearch
from langchain.tools import Tool
from dotenv import load_dotenv

load_dotenv(override=True)

def search_quarterly(year: int = None, quarter: int = None) -> str:
    """Search for NVIDIA quarterly report information online."""
    # Construct search query
    query = f"NVIDIA Q{quarter} {year} quarterly earnings report"
    
    try:
        # Use SerpAPI for web search
        search = GoogleSearch({
            "q": query,
            "api_key": os.getenv("SERP_API_KEY"),
            "num": 5
        })
        results = search.get_dict()
        
        # Extract and summarize results
        summary = "Web Search Results:\n\n"
        
        if "organic_results" in results and results["organic_results"]:
            for i, result in enumerate(results["organic_results"][:3], 1):
                summary += f"{i}. {result.get('title', 'No title')}\n"
                summary += f"   {result.get('snippet', 'No snippet')}\n"
                summary += f"   URL: {result.get('link', 'No link')}\n\n"
        else:
            summary += "No search results found."
            
        return summary
    
    except Exception as e:
        return f"Error performing web search: {str(e)}\n\nSimulated results for NVIDIA Q{quarter} {year}:\n" + \
               "- Strong revenue growth in data center segment\n" + \
               "- AI chip demand continues to drive performance\n" + \
               "- Gaming revenue shows slight recovery\n"

# Create LangChain tool
web_search_tool = Tool(
    name="nvidia_web_search",
    description="Search the web for NVIDIA quarterly report information",
    func=search_quarterly
)