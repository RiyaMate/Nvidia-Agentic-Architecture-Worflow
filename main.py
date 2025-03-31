from fastapi import FastAPI, HTTPException
from typing import List, Dict, Any
from pydantic import BaseModel
import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '.')))
from langGraph.pipeline import build_pipeline

# Initialize FastAPI app
app = FastAPI()

class QueryRequest(BaseModel):
    """Request model for research report endpoint"""
    question: str
    search_type: str
    selected_periods: List[str]
    agents: List[str]
    model: str = "claude-3-haiku-20240307"

    class Config:
        json_schema_extra = {
            "example": {
                "question": "Analyze NVIDIA's performance",
                "search_type": "Specific Quarter",
                "selected_periods": ["2023q4"],
                "agents": ["RAG Agent", "Web Search Agent"],
                "model": "claude-3-haiku-20240307"
            }
        }

@app.post("/research_report")
async def research_report(request: QueryRequest):
    """Generate research report based on query and selected agents"""
    try:
        # Create initial state for pipeline
        state = {
            "input": request.question,
            "question": request.question,
            "search_type": request.search_type,
            "selected_periods": request.selected_periods,
            "chat_history": [],
            "intermediate_steps": [],
            "selected_agents": request.agents,
            "model": request.model,
        }

        # Initialize pipeline with selected agents
        pipeline = build_pipeline(selected_agents=request.agents,model=request.model)
        
        # Execute pipeline
        result = pipeline.invoke(state)
        
        if not result:
            raise HTTPException(
                status_code=500,
                detail="Pipeline execution failed to produce results"
            )
        if isinstance(result.get("final_report"), str):
            result["final_report"] = result["final_report"].replace("\\n", "\n")
        
        if "web_links" not in result and "web_links"  in state:
            result["web_links"] = state["web_links"]
        if "web_images" not in result and "web_images" in state:
            result["web_images"] = state["web_images"]
            
        return result
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Analysis failed: {str(e)}"
        )

