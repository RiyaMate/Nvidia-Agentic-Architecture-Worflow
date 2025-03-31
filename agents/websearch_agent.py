import os
from typing import List, Optional
from serpapi import GoogleSearch
from langchain.tools import Tool
from dotenv import load_dotenv

load_dotenv(override=True)

def search_quarterly(query: Optional[str] = None, selected_periods: Optional[List[str]] = None) -> dict:
    """
    Search for NVIDIA quarterly report information online.
    Returns structured data including text summary, image URLs, and links.
    """
    # Build search query
    if not query:
        query = "NVIDIA financial performance"
        
    search_query = query
    
    if selected_periods and selected_periods[0] != "all":
        # Parse each period and add to query
        parsed = []
        for period in selected_periods:
            try:
                year = period[:4]
                quarter_num = period[4:].replace("q", "")
                parsed.append(f"Q{quarter_num} {year}")
            except:
                parsed.append(period)
        combined_str = ", ".join(parsed)
        search_query = f"{query} for the periods of {combined_str}"

    try:
        # Store results
        summary = f"=== Web Search for: '{search_query}' ===\n\n"
        links = []
        images = []
        
        # Text search
        text_params = {
            "q": search_query,
            "api_key": os.getenv("SERP_API_KEY"),
            "num": 5
        }
        text_search = GoogleSearch(text_params)
        text_results = text_search.get_dict()
        
        # Process organic results
        if "organic_results" in text_results and text_results["organic_results"]:
            orgs = text_results["organic_results"]
            for i, result in enumerate(orgs[:3], 1):
                title = result.get('title', 'No title')
                snippet = result.get('snippet', 'No snippet')
                link = result.get('link', 'No link')
                links.append(link)
                summary += f"{i}. {title}\n   {snippet}\n   URL: {link}\n\n"
        else:
            summary += "No textual search results found.\n\n"
        
        # Image search
        image_params = {
            "q": search_query,
            "api_key": os.getenv("SERP_API_KEY"),
            "tbm": "isch",
            "num": 5
        }
        image_search = GoogleSearch(image_params)
        image_results = image_search.get_dict()
        
        summary += "=== Image Search Results ===\n"
        
        if "images_results" in image_results and image_results["images_results"]:
            imgs = image_results["images_results"]
            for i, img_item in enumerate(imgs[:3], 1):
                thumbnail = img_item.get("thumbnail", "")
                title = img_item.get("title", "No title")
                
                if thumbnail:
                    images.append(thumbnail)
                
                summary += f"{i}. {title}\n"
                summary += f"   Thumbnail: {thumbnail}\n\n"
        
        # Return structured results
        return {
            "text": summary,
            "links": links,
            "images": images
        }
    
    except Exception as e:
        return {
            "text": f"Error performing web search: {str(e)}\n\nSimulated fallback:\n- Strong revenue growth in data center segment\n- AI chip demand continues to drive performance",
            "links": [],
            "images": []
        }
    
if __name__ == "__main__":
    # Test the web search tool
    result = search_quarterly("what is the nvidia performance",["2023q1", "2023q2"])
    print(result["links"])
    print("_____images_____")
    print(result["images"])
    # print(search_quarterly())