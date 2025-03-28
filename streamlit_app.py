import streamlit as st
import requests
import json
import toml
import os
# -------------------------------
# 1) Page Config for Wide Layout
# -------------------------------
st.set_page_config(
    page_title="NVIDIA Research Assistant",
    layout="wide",
    initial_sidebar_state="expanded"
)

# -------------------------------
# 2) Configuration
# -------------------------------
config = toml.load("config.toml")
API_URL = config["fastapi_url"]
QUERY_URL = f"{API_URL}/research_report"

# -------------------------------
# 3) Helper Functions
# -------------------------------
def display_financial_data(data):
    """Display Snowflake financial metrics and chart."""
    if not data:
        st.info("No financial data available")
        return

    st.markdown("### NVIDIA Financial Metrics")
    if "chart" in data:
        st.image(
            f"data:image/png;base64,{data['chart']}",
            caption="NVIDIA Valuation Metrics",
            use_column_width=True
        )
    if "metrics" in data:
        st.markdown("#### Key Metrics")
        if isinstance(data["metrics"], list):
            st.dataframe(data["metrics"])
        else:
            st.write(data["metrics"])


def display_rag_results(data):
    """Display results from the RAG Agent."""
    if not data:
        st.info("No document analysis available")
        return

    st.markdown("### Document Analysis Results")
    st.markdown(data.get("result", "No results found"))

    if "sources" in data:
        with st.expander("📚 Source Documents"):
            for src in data["sources"]:
                st.markdown(f"- {src}")


def display_web_results(data):
    """Display results from the Web Search Agent."""
    if not data:
        st.info("No web search results available")
        return

    st.markdown("### Web Search Results")
    st.markdown(data)

# -------------------------------
# 4) Sidebar Configuration
# -------------------------------
st.sidebar.title("NVIDIA Research Assistant")
st.sidebar.markdown("### Search Configuration")

search_type = st.sidebar.radio(
    "Select Search Type",
    ["All Quarters", "Specific Quarter"],
    key="search_type"
)

# Keep user’s selected periods in session
if "selected_periods" not in st.session_state:
    st.session_state.selected_periods = ["2023q1"]

if search_type == "Specific Quarter":
    all_periods = [f"{y}q{q}" for y in range(2020, 2026) for q in range(1, 5)]
    # 1) Filter out "all" from the default to avoid the Streamlit error
    default_selected = [
        p for p in st.session_state.selected_periods
        if p in all_periods
    ]
    if not default_selected:
        default_selected = ["2023q1"]  # Some safe default

    selected_periods = st.sidebar.multiselect(
        "Select Period(s)",
        options=all_periods,
        default=default_selected,  # Use the filtered list
        key="period_select"
    )

    if not selected_periods:
        selected_periods = ["2023q1"]
    st.session_state.selected_periods = selected_periods

else:
    selected_periods = ["all"]
    st.session_state.selected_periods = selected_periods


st.sidebar.markdown("---")
st.sidebar.markdown("### Agent Configuration")

if "selected_agents" not in st.session_state:
    st.session_state.selected_agents = ["RAG Agent"]

available_agents = ["RAG Agent", "Web Search Agent", "Snowflake Agent"]
selected_agents = st.sidebar.multiselect(
    "Select AI Agents (at least one required)",
    options=available_agents,
    default=st.session_state.selected_agents,
    key="agent_select_unique"
)

# Validate agent selection
if not selected_agents:
    st.sidebar.warning("⚠️ At least one agent is required")
    selected_agents = ["RAG Agent"]
st.session_state.selected_agents = selected_agents.copy()

if st.sidebar.button("Apply Agent Selection", type="primary", use_container_width=True, key="apply_agents_unique"):
    st.session_state.selected_agents = selected_agents.copy()
    st.sidebar.success("✅ Agent selection updated!")

# create model selection for the user
st.sidebar.markdown("---")
st.sidebar.markdown("### Model Configuration")
available_models = {
    "Claude 3 Haiku": "claude-3-haiku-20240307", 
    "Claude 3 Sonnet": "claude-3-5-sonnet-20240620",
    "Gemini Pro": "gemini-2.0-flash",
    "DeepSeek": "deepseek-reasoner",
    "Grok-1": "grok-2-latest"
}
if "selected_model" not in st.session_state:
    st.session_state.selected_model = "Claude 3 Haiku"
# Create a selection box for model selction
model_display_name = st.sidebar.selectbox(
    "Select Model",
    options=list(available_models.keys()),
    index=list(available_models.keys()).index(st.session_state.selected_model)
        if st.session_state.selected_model in available_models.keys() else 0,
    key="model_select"
)
if st.session_state.selected_model != model_display_name:
    st.session_state.selected_model = model_display_name
    st.session_state.model_changed = True

# Add apply button for model selection
if st.sidebar.button("Apply Model Selection", type="primary", use_container_width=True, key="apply_model"):
    st.sidebar.success(f"✅ Model updated to {model_display_name}!")
    # Force rerun for immediate UI update
    st.rerun()

# Add model description based on selection
model_descriptions = {
    "Claude 3 Haiku": "Fast, compact model with strong reasoning capabilities (Anthropic)",
    "Claude 3 Sonnet": "Balanced performance with enhanced reasoning (Anthropic)", 
    "Gemini Pro": "Google's advanced model with strong coding abilities (Google)",
    "DeepSeek": "Specialized for code generation and technical tasks (DeepSeek)",
    "Grok-1": "Conversational model focused on insightful responses (xAI)"
}

with st.sidebar.expander("📝 Model Info"):
    st.markdown(f"**{model_display_name}**")
    st.markdown(model_descriptions.get(model_display_name, "No description available"))

# -------------------------------
# 5) Navigation
# -------------------------------
if "current_page" not in st.session_state:
    st.session_state.current_page = "Home"
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

home_btn = st.sidebar.button("Home", key="nav_Home", use_container_width=True)
report_btn = st.sidebar.button("Combined Report", key="nav_Report", use_container_width=True)
about_btn = st.sidebar.button("About", key="nav_About", use_container_width=True)

if home_btn:
    st.session_state.current_page = "Home"
    st.rerun()
elif report_btn:
    st.session_state.current_page = "Combined Report"
    st.rerun()
elif about_btn:
    st.session_state.current_page = "About"
    st.rerun()

page = st.session_state.current_page

# -------------------------------
# 6) Page Layout
# -------------------------------
if page == "Home":
    st.title("Welcome to the NVIDIA Multi-Agent Research Assistant")
    st.markdown("""
        ## Generate comprehensive financial analysis reports on NVIDIA
        
        This application combines multiple specialized data sources:
        
        
        - **📊 Financial Reports**: Historical quarterly reports with detailed metrics from Pinecone
        - **🔍 Web Research**: Real-time market insights and news from SerpAPI
        - **📈 Financial Analysis**: Structured valuation metrics and visualization from Snowflake
        
        ### How to use:
        1. Select search periods in the sidebar (All quarters or specific quarters)
        2. Choose which AI agents to activate for your report
        3. Navigate to the "Combined Report" page
        4. Enter your research question
        5. Receive a comprehensive, structured analysis report
        
        ### Sample questions:
        - "How has NVIDIA's revenue growth been trending in Q4 2023?"
        - "What were the key drivers of NVIDIA's profitability in 2023?"
        - "Analyze NVIDIA's data center segment performance in recent quarters"
        - "How do NVIDIA's valuation metrics compare to industry standards?"
    """)
    st.image("nvidia_workflow.png", caption="NVIDIA Multi-Agent Workflow Architecture", use_column_width=True)
    

elif page == "Combined Report":
    st.title("NVIDIA Research Assistant")
    st.subheader("💬 Research History")

    # # Show chat history
    # with st.container():
    #     for message in st.session_state.chat_history:
    #         if message["role"] == "user":
    #             periods_text = ", ".join(message.get("selected_periods", []))
    #             agents_text = ", ".join(message.get("agents", []))
    #             st.markdown(f"""
    #                 <div class='user-message'>
    #                     <div class='metadata'>📅 {periods_text}<br>🤖 Agents: {agents_text}</div>
    #                     <div>🔍 {message['content']}</div>
    #                 </div>
    #             """, unsafe_allow_html=True)
    #         else:
    #             st.markdown(f"""
    #                 <div class='assistant-message'>
    #                     <div class='metadata'>🤖 NVIDIA Research Assistant</div>
    #                     <div>{message['content']}</div>
    #                 </div>
    #             """, unsafe_allow_html=True)

    st.markdown("---")
    with st.form("report_form", clear_on_submit=False):
        col1, col2 = st.columns([3, 1])
        with col1:
            question = st.text_input("Research Question", placeholder="What has driven NVIDIA's revenue growth?")
        with col2:
            submitted = st.form_submit_button("➤")
    result_container = st.container()

    if submitted and question:
        if "processing" not in st.session_state:
            st.session_state.processing = True

        with st.spinner("🔄 Generating report..."):
            st.info("📊 Analyzing NVIDIA data from multiple sources...")
            payload = {
                "question": question,
                "search_type": search_type,
                "selected_periods": selected_periods,
                "agents": selected_agents,
                "model": available_models[st.session_state.selected_model]
            }
            print(f"using model: {st.session_state.selected_model} ({available_models[st.session_state.selected_model]})")
            try:
                response = requests.post(QUERY_URL, json=payload)
                if response.status_code == 200:
                    data = response.json()
                    with result_container:
                        if "final_report" in data:
                            st.session_state.current_report = data
                            used_model = st.session_state.selected_model
                            st.markdown(f"**Analysis by {used_model}**")
                            st.markdown(data.get("final_report", ""), unsafe_allow_html=True)
                            st.markdown("---")
                            st.subheader("📊 Detailed Results")
                            tabs_to_show = ["Overview"]
                            if "Snowflake Agent" in selected_agents:
                                tabs_to_show.append("Financial Data")
                            if "RAG Agent" in selected_agents:
                                tabs_to_show.append("Document Analysis")
                            if "Web Search Agent" in selected_agents:
                                tabs_to_show.append("Web Results")
                            tab_objects = st.tabs(tabs_to_show)
                            for i,tab_name in enumerate(tabs_to_show):
                                with tab_objects[i]:
                                    print(f"using model: {st.session_state.selected_model} ({available_models[st.session_state.selected_model]})")

                                    # In streamlit_app.py, update the Overview tab handling:
                                    if tab_name == "Overview":
                                        # Print debug info to check what data is available
                                        st.write(f"Data keys: {list(data.keys())}")
                                        
                                        # Show introduction
                                        intro = data.get("introduction", "")
                                        if intro and len(intro) > 10:
                                            st.markdown(intro)
                                        
                                        # Show executive summary with proper error handling
                                        exec_summary = data.get("executive_summary", "")
                                        if exec_summary and len(exec_summary) > 10:
                                            st.markdown(exec_summary)
                                        else:
                                            # Try to use part of the final report
                                            final_report = data.get("final_report", "")
                                            if final_report:
                                                sections = final_report.split("##")
                                                if len(sections) > 1:
                                                    # The second section is typically the executive summary
                                                    st.markdown("## Executive Summary")
                                                    st.markdown(sections[1])
                                                else:
                                                    st.info("No executive summary available")
                                        
                                        # Show conclusion
                                        conclusion = data.get("conclusion", "")
                                        if conclusion and len(conclusion) > 10:
                                            st.markdown(conclusion)
                                                                                                                
                                    elif tab_name == "Document Analysis":
                                        # First try using historical_data
                                        historical_data = data.get("historical_data", "")
                                        
                                        if historical_data and "No historical" not in historical_data and len(historical_data) > 50:
                                            st.markdown(historical_data)
                                        else:
                                            # Then try using RAG summary directly as fallback
                                            rag_summary = data.get("rag_summary", "")
                                            if rag_summary and "No historical" not in rag_summary and len(rag_summary) > 30:
                                                st.markdown("### Historical Data from Financial Reports")
                                                st.markdown(rag_summary)
                                            else:
                                                # Finally, check if there's any RAG output directly
                                                if "rag_output" in data and isinstance(data["rag_output"], dict):
                                                    rag_result = data["rag_output"].get("result", {})
                                                    if isinstance(rag_result, dict):
                                                        rag_insights = rag_result.get("insights", "")
                                                        if rag_insights:
                                                            st.markdown("### Historical Data from Financial Reports")
                                                            st.markdown(rag_insights)
                                                        else:
                                                            st.info("No historical financial data available for this query.")
                                                else:
                                                    st.info("No historical financial data available for this query.")
                                                    
                                        # Add instructions to get better results
                                        if "RAG Agent" not in selected_agents:
                                            st.warning("💡 To see historical data, enable the RAG Agent in the sidebar")

                                    elif tab_name == "Web Results":
                                        web_insights = data.get("web_insights", "")
                                        if web_insights and len(web_insights) > 50:
                                            st.markdown(web_insights)
                                        else:
                                            # Try to use web_summary as fallback
                                            web_summary = data.get("web_summary", "")
                                            if web_summary and "No web data" not in web_summary:
                                                st.markdown("### Web Search Summary")
                                                st.markdown(web_summary)
                                            else:
                                                st.info("No web insights available for this query.")
                                        
                                        # Display web images
                                        web_images = data.get("web_images", [])
                                        if web_images:
                                            st.subheader("📊 Web Search Images")
                                            cols = st.columns(min(3, len(web_images)))
                                            for i, img_url in enumerate(web_images[:3]):
                                                try:
                                                    cols[i].image(img_url, width=200)
                                                except:
                                                    cols[i].error("Image load failed")
                                        
                                        # Display web links
                                        web_links = data.get("web_links", [])
                                        if web_links:
                                            st.subheader("🔗 Web Sources")
                                            for i, link in enumerate(web_links):
                                                try:
                                                    domain = link.split('/')[2]
                                                except:
                                                    domain = link
                                                st.markdown(f"{i+1}. [{domain}]({link})")
                                        else:
                                            st.info("No web links available")
                                            
                                        # Add instructions to get better results
                                        if "Web Search Agent" not in selected_agents:
                                            st.warning("💡 To see web research results, enable the Web Search Agent in the sidebar")
                        else:
                            st.error("❌ No report generated. Please check the API response.")

                    
                else:
                    st.error(f"❌ API Error: {response.status_code} - {response.text}")
            except Exception as e:
                st.error(f"❌ Error: {str(e)}")
            st.session_state.processing = False

    elif page == "About":
        st.title("About NVIDIA Research Assistant")
        st.markdown("""
            **NVIDIA Multi-Agent Research Assistant** integrates:
            - **RAG Agent**: Uses Pinecone with metadata filtering to retrieve historical reports.
            - **Web Search Agent**: Uses SerpAPI for real-time search.
            - **Snowflake Agent**: Connects to Snowflake for valuation measures and charts.
        """)

# -------------------------------
# 7) Custom CSS (UNCHANGED)
# -------------------------------
st.markdown("""
<style>
/* ---------------------------------- */
/* Dark background, Nvidia green accent */
/* ---------------------------------- */
body, .main, [data-testid="stHeader"], [data-testid="stSidebar"] {
    background-color: #1E1E1E !important;
    color: #FFFFFF !important;
    font-family: "Segoe UI", sans-serif;
}

/* ---------------------------------- */
/* Make links NVIDIA green, no underline */
/* ---------------------------------- */
a, a:visited {
    color: #76B900 !important; /* Nvidia green */
    text-decoration: none !important;
}
a:hover {
    color: #5c8d00 !important; 
    text-decoration: underline !important;
}

/* ---------------------------------- */
/* Headings in NVIDIA green */
/* ---------------------------------- */
h1, h2, h3, h4 {
    color: #76B900 !important; /* Nvidia Green */
}

/* ---------------------------------- */
/* Block container width */
/* ---------------------------------- */
.block-container {
    max-width: 1400px; /* Full width container */
}

/* ---------------------------------- */
/* Chat Bubbles styling */
/* ---------------------------------- */
.chat-container {
    margin-bottom: 30px;
    max-height: 55vh; /* adjustable height */
    overflow-y: auto;
    padding: 1em;
    border: 1px solid #3a3a3a;
    border-radius: 10px;
    background-color: #2b2b2b;
}
.user-message {
    background-color: #2E8B57; /* or #2196F3 if you prefer bluish */
    padding: 15px;
    border-radius: 15px;
    margin: 10px 0;
    color: white;
}
.assistant-message {
    background-color: #262730;
    padding: 15px;
    border-radius: 15px;
    margin: 10px 0;
    color: white;
}
.metadata {
    font-size: 0.8em;
    color: #B0B0B0;
    margin-bottom: 5px;
}

/* ---------------------------------- */
/* Circular submit button (ChatGPT style) */
/* ---------------------------------- */
[data-testid="stFormSubmitButton"] button {
    border-radius: 50%;
    width: 50px;
    height: 50px;
    padding: 0;
    min-width: 0;
    font-size: 1.4em;
    font-weight: bold;
    background-color: #76B900;
    color: #fff;
    border: none;
    transition: background-color 0.3s ease;
}
[data-testid="stFormSubmitButton"] button:hover {
    background-color: #5c8d00;
}

/* ---------------------------------- */
/* Tab styling to match dark theme */
/* ---------------------------------- */
div[data-testid="stTabs"] button {
    background-color: #333333;
    color: #fff;
    border: none;
    border-radius: 0;
    padding: 0.5rem 1rem;
}
div[data-testid="stTabs"] button[aria-selected="true"] {
    background-color: #76B900 !important;
    color: #000 !important;
    font-weight: bold;
}

/* ---------------------------------- */
/* Datatable override for dark theme */
/* ---------------------------------- */
[data-testid="stDataFrame"] {
    background-color: #262730;
    color: #fff;
    border: none;
}
[data-testid="stDataFrame"] table {
    color: #fff;
}

/* ---------------------------------- */
/* Sidebar background & text colors */
/* ---------------------------------- */
[data-testid="stSidebar"] {
    background-color: #2b2b2b !important; /* Dark gray background */
}
[data-testid="stSidebar"] label, 
[data-testid="stSidebar"] .stRadio,
[data-testid="stSidebar"] .stButton {
    color: #fff !important; /* White text */
}
[data-testid="stSidebar"] .stRadio > div:hover {
    background-color: #333333 !important;
}

/* -------------------------------------- */
/* Sidebar nav buttons in NVIDIA green   */
/* -------------------------------------- */
[data-testid="stSidebar"] .stButton > button {
    background-color: #0a8006 !important; /* NVIDIA green */
    color: #fff !important;
    border: none !important;
}
[data-testid="stSidebar"] .stButton > button:hover {
    background-color: #5c8d00 !important; /* darker green hover */
    color: #fff !important;
}

/* ---------------------------------- */
/* Agent Selection Submit Button */
/* ---------------------------------- */
[data-testid="stButton"] button[kind="primary"] {
    
    color: white !important;
    border: none !important;
    padding: 0.5rem 1rem !important;
    border-radius: 4px !important;
    transition: background-color 0.3s ease !important;
}

[data-testid="stButton"] button[kind="primary"]:hover {
    background-color: #5c8d00 !important;
}

/* ---------------------------------- */
/* Navigation Buttons Layout */
/* ---------------------------------- */
[data-testid="column"] {
    padding: 0.25rem !important;
}

[data-testid="stButton"] button {
    width: 100% !important;
    margin: 0 !important;
}
/* ---------------------------------- */
/* Improved Markdown styling */
/* ---------------------------------- */
.stMarkdown h1 {
    color: #76B900 !important;
    border-bottom: 2px solid #76B900;
    padding-bottom: 0.3em;
    margin-bottom: 0.8em;
}

.stMarkdown h2 {
    color: #76B900 !important;
    border-bottom: 1px solid #444;
    padding-bottom: 0.2em;
    margin-top: 1.5em;
    margin-bottom: 0.8em;
}

.stMarkdown h3 {
    color: #76B900 !important;
    margin-top: 1.2em;
    margin-bottom: 0.6em;
}

.stMarkdown ul, .stMarkdown ol {
    margin-bottom: 1em;
    padding-left: 1.5em;
}

.stMarkdown li {
    margin-bottom: 0.3em;
}

.stMarkdown p {
    margin-bottom: 1em;
    line-height: 1.5;
}

.stMarkdown blockquote {
    border-left: 4px solid #76B900;
    padding-left: 1em;
    margin-left: 0;
    color: #ccc;
}

.stMarkdown table {
    width: 100%;
    border-collapse: collapse;
    margin: 1em 0;
}

.stMarkdown th, .stMarkdown td {
    padding: 8px 12px;
    border: 1px solid #444;
    text-align: left;
}

.stMarkdown th {
    background-color: #333;
    color: white;
    font-weight: bold;
}

.stMarkdown tr:nth-child(even) {
    background-color: #2a2a2a;
}

/* Remove chat history styling since we're not using it */
.user-message, .assistant-message {
    display: none;
}
/* ---------------------------------- */
/* NVIDIA-themed Radio Buttons */
/* ---------------------------------- */
[data-testid="stRadio"] label {
    color: white !important;
}

/* Selected radio button */
[data-testid="stRadio"] div[role="radiogroup"] label[data-baseweb="radio"] div:first-child div {
    background-color: #76B900 !important; /* NVIDIA green for selected radio */
    border-color: #76B900 !important;
}

/* Unselected radio button hover */
[data-testid="stRadio"] div[role="radiogroup"] label[data-baseweb="radio"] div:first-child:hover div {
    border-color: #76B900 !important;
}

/* ---------------------------------- */
/* NVIDIA-themed Multiselect */
/* ---------------------------------- */
/* Multiselect container */
[data-testid="stMultiSelect"] div[data-baseweb="select"] {
    background-color: #333333 !important;
    border-color: #555555 !important;
}

/* Dropdown menu background */
div[role="listbox"] {
    background-color: #333333 !important;
    border-color: #555555 !important;
}

/* Selected items */
[data-testid="stMultiSelect"] div[data-baseweb="tag"] {
    background-color: #76B900 !important; /* NVIDIA green */
    color: black !important;
    font-weight: 500 !important;
}

/* Delete button (x) in selected tags */
[data-testid="stMultiSelect"] div[data-baseweb="tag"] span:last-child {
    color: black !important;
}

/* Hover state for options */
div[role="option"]:hover {
    background-color: rgba(118, 185, 0, 0.3) !important; /* Semi-transparent NVIDIA green */
}

/* Selected option in dropdown */
div[aria-selected="true"] {
    background-color: rgba(118, 185, 0, 0.5) !important; /* Semi-transparent NVIDIA green */
}

/* Dropdown option text color */
div[role="option"] {
    color: white !important;
}

/* Dropdown icon color */
[data-testid="stMultiSelect"] div[role="combobox"] svg {
    color: #76B900 !important;
}
.st-d5 .st-bp {
    background: aliceblue;
    color: #76B900 !important;
}

/* Dropdown border when focused */
[data-testid="stMultiSelect"] div[data-baseweb="select"]:focus-within {
    border-color: #76B900 !important;
    box-shadow: 0 0 0 1px #76B900 !important;
}

/* Remove item hover effect */
[data-testid="stMultiSelect"] div[data-baseweb="tag"]:hover {
    background-color: #5c8d00 !important; /* Darker NVIDIA green */
}

/* ---------------------------------- */
/* Override Default Font Colors */
/* ---------------------------------- */
label {
    color: white !important;
}

/* Remove icon colors */
[data-testid="stMultiSelect"] span[aria-hidden="true"] svg {
    color: black !important;
}

/* Make the placeholder white */
[data-testid="stMultiSelect"] [data-baseweb="select"] input::placeholder {
    color: rgba(255, 255, 255, 0.7) !important;
}

/* Input text color */
[data-testid="stMultiSelect"] input {
    color: white !important;
}
            
.stFormSubmitButton.st-emotion-cache-8atqhb.e1mlolmg0 {
    /* position: relative; */
    margin-top: 20px;
    margin-left: 31px;
    }
.st-f2{
    border-color: black;}
            
/* ---------------------------------- */
/* Custom CSS for better loading experience */
.stSpinner > div {
    border-color: #76B900 !important;
    border-top-color: transparent !important;
}
.stSpinner {
    margin-bottom: 1rem !important;
}

/* Info box styling */
.stAlert [data-testid="stInfoBox"] {
    background-color: rgba(118, 185, 0, 0.1) !important;
    color: #FFFFFF !important;
    border-color: #76B900 !important;
}

/* Persistent styling for transitions */
[data-testid="stForm"] {
    background-color: #2b2b2b !important;
    padding: 1rem !important;
    border-radius: 0.5rem !important;
    margin-bottom: 1rem !important;
}

/* Fix for text inputs during transitions */
input[type="text"] {
    background-color: #333333 !important;
    color: white !important;
    border-color: #555555 !important;
}

/* Fix for input focus state */
input[type="text"]:focus {
    border-color: #76B900 !important;
    box-shadow: 0 0 0 1px #76B900 !important;
}

/* Ensure tabs retain styling during loading */
[data-testid="stTabs"] {
    background-color: #1E1E1E !important;
}
/* Image styling for financial charts */
[data-testid="stImage"] {
    border: 1px solid #444;
    border-radius: 5px;
    padding: 10px;
    background-color: #2a2a2a;
    box-shadow: 0 4px 6px rgba(0, 0, 0, 0.3);
    margin: 10px 0;
}

[data-testid="stImage"] > img {
    max-width: 100%;
    height: auto;
    display: block;
    margin: 0 auto;
}

/* Caption styling */
[data-testid="stImage"] > div {
    text-align: center;
    color: #76B900 !important; /* NVIDIA green for captions */
    padding-top: 10px;
    font-weight: 600;
    font-style: italic;
}
/* ---------------------------------- */
/* submit button hovering effect*/
/* ---------------------------------- */
.st-emotion-cache-b0y9n5:hover {
    background-color: #5c8d00 !important; /* Darker NVIDIA green */
    color: white !important;
}
.st-emotion-cache-b0y9n5:active {
    background-color: #76B900 !important; /* NVIDIA green */
    color: white !important;
}
.st-emotion-cache-b0y9n5:focus-visible {
    outline: none !important;
    box-shadow: 0 0 0 2px #76B900 !important; /* NVIDIA green focus outline */
    border-radius: 4px !important;
    background-color: #76B900 !important; /* NVIDIA green */
}
.st-emotion-cache-b0y9n5:focus:not(:active){
            
            background-color: #76B900 !important; /* NVIDIA green */
            }
.st-av{
            
            background-color: #76B900 !important;
    border-color: #0a8006;
            }
.stRadio > .st-bp{
            background: none
            }
.st-emotion-cache-4rp1ik:hover,
            .st-emotion-cache-4rp1ik:hover svg{
            color: #5c8d00}
    

</style>
""", unsafe_allow_html=True)

st.markdown("""
<script>
// This script helps maintain styles during transitions
const observer = new MutationObserver((mutations) => {
  mutations.forEach((mutation) => {
    if (mutation.addedNodes.length) {
      mutation.addedNodes.forEach((node) => {
        if (node.nodeType === 1) { // Element node
          // Restore NVIDIA styling to radio buttons
          const radios = node.querySelectorAll('[role="radiogroup"] [data-baseweb="radio"] div:first-child div');
          radios.forEach(radio => {
            if (radio.getAttribute('aria-checked') === 'true') {
              radio.style.backgroundColor = '#76B900';
              radio.style.borderColor = '#76B900';
            }
          });
          
          // Restore NVIDIA styling to multiselect
          const tags = node.querySelectorAll('[data-baseweb="tag"]');
          tags.forEach(tag => {
            tag.style.backgroundColor = '#76B900';
            tag.style.color = 'black';
            tag.style.fontWeight = '500';
          });
          
          // Restore text and background colors
          const elements = node.querySelectorAll('*');
          elements.forEach(el => {
            if (window.getComputedStyle(el).backgroundColor === 'rgb(255, 255, 255)') {
              el.style.backgroundColor = '#1E1E1E';
            }
          });
        }
      });
    }
  });
});

// Start observing the document
observer.observe(document.body, {
  childList: true,
  subtree: true
});
</script>
""", unsafe_allow_html=True)
