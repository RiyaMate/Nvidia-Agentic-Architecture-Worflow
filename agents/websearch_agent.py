import os
from typing import List, Optional
from serpapi import GoogleSearch
from langchain.tools import Tool
from dotenv import load_dotenv

load_dotenv(override=True)

def search_quarterly(query: Optional[str] = None, selected_periods: Optional[List[str]] = None) -> str:
    """
    Search for NVIDIA quarterly report information online.
    Args:
        query: The user's specific query about NVIDIA (e.g., "revenue growth")
        selected_periods: List of periods (e.g., ["2023q1","2023q2"])
    
    If no periods are provided, searches 'all-time' across all quarters.
    If no query is provided, searches for general quarterly reports.
    Also performs an image search, and appends those results at the end.
    """
    # Base query - either user specified or default
    
    
    if not selected_periods:
        # Default: generic query for all-time
        search_query = query
    else:
        # Parse each period string (e.g., "2023q1") => "Q1 2023"
        parsed = []
        for period in selected_periods:
            try:
                year = period[:4]
                quarter_num = period[4:].replace("q", "")  # e.g. "q1" => "1"
                parsed.append(f"Q{quarter_num} {year}")
            except:
                # If parsing fails, fallback to the raw period
                parsed.append(period)
        # Join them into a single query phrase
        combined_str = ", ".join(parsed)
        search_query = f"{query} for the periods of {combined_str}"

    try:
        # --- 1) TEXT SEARCH ---
        text_params = {
            "q": search_query,
            "api_key": os.getenv("SERP_API_KEY"),
            "num": 5
        }
        text_search = GoogleSearch(text_params)
        text_results = text_search.get_dict()
        
        summary = f"=== Web Search for: '{search_query}' ===\n\n"
        
        # Process textual/organic results
        if "organic_results" in text_results and text_results["organic_results"]:
            orgs = text_results["organic_results"]
            for i, result in enumerate(orgs[:3], 1):
                title = result.get('title', 'No title')
                snippet = result.get('snippet', 'No snippet')
                link = result.get('link', 'No link')
                summary += f"{i}. {title}\n   {snippet}\n   URL: {link}\n\n"
        else:
            summary += "No textual search results found.\n\n"
        
        # --- 2) IMAGE SEARCH ---
        image_params = {
            "q": search_query,
            "api_key": os.getenv("SERP_API_KEY"),
            "tbm": "isch",  # 'image search' mode
            "num": 5
        }
        image_search = GoogleSearch(image_params)
        image_results = image_search.get_dict()
        
        summary += "=== Image Search Results ===\n"
        
        if "images_results" in image_results and image_results["images_results"]:
            imgs = image_results["images_results"]
            for i, img_item in enumerate(imgs[:3], 1):
                link = img_item.get("link", "No link")
                thumbnail = img_item.get("thumbnail", "No thumbnail")
                title = img_item.get("title", "No title")
                
                summary += f"{i}. {title}\n"
                summary += f"   Link: {link}\n"
                summary += f"   Thumbnail: {thumbnail}\n\n"
        else:
            summary += "No image results found.\n"
        
        return summary
    
    except Exception as e:
        return (
            f"Error performing web search: {str(e)}\n\n"
            f"Simulated fallback:\n"
            "- Strong revenue growth in data center segment\n"
            "- AI chip demand continues to drive performance\n"
            "- Gaming revenue shows slight recovery\n"
        )

if __name__ == "__main__":
    # Test the web search tool
    print(search_quarterly("what is the nvidia performance",["2023q1", "2023q2"]))
    # print(search_quarterly())