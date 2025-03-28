import streamlit as st
import requests
import json
import base64
import toml

# -------------------------------------
# 1) Set Page Config for Wide Layout
# -------------------------------------
st.set_page_config(
    page_title="NVIDIA Research Assistant",
    layout="wide",
    initial_sidebar_state="expanded"
)

# -------------------------------
# Configuration
# -------------------------------
config = toml.load("config.toml")
API_URL = config["fastapi_url"]
QUERY_URL = f"{API_URL}/research_report"

# -------------------------------
# Sidebar Navigation
# -------------------------------
st.sidebar.title("NVIDIA Research Assistant")

# Search Type and Agent Selection
st.sidebar.markdown("### Search Configuration")

# Update the search type radio button
search_type = st.sidebar.radio(
    "Select Search Type",
    options=["All Quarters", "Specific Quarter"],
    key="search_type"
)

if search_type == "Specific Quarter":
    # Generate all year-quarter combinations from 2020q1 to 2025q4
    quarter_options = [
        f"{year}q{quarter}" for year in range(2020, 2026)
        for quarter in range(1, 5)
    ]
    selected_periods = st.sidebar.multiselect(
        "Select Period(s)",
        options=quarter_options,
        default=["2023q1"],
        key="period_select"
    )
    if not selected_periods:
        selected_periods = ["2023q1"]
    # Keep these in session_state for later usage
    st.session_state.year_slider = selected_periods[0].split('q')[0]
    st.session_state.quarter_slider = selected_periods[0].split('q')[1]
else:
    st.session_state.year_slider = "all"
    st.session_state.quarter_slider = "all"
    st.session_state.period_select = ["all"]

# Add spacing in sidebar
st.sidebar.markdown("---")

# Add Agent Selection after periods
st.sidebar.markdown("### Agent Configuration")

# multiselect for agents
available_agents = ["Snowflake Agent", "RAG Agent", "Web Search Agent"]
selected_agents = st.sidebar.multiselect(
    "Select AI Agents",
    options=available_agents,
    default=available_agents,  # By default, select all agents
    key="agent_type"
)

# Add an "All Agents" checkbox
use_all_agents = st.sidebar.checkbox(
    "Use All Agents",
    value=True,
    key="use_all_agents"
)

# Add a submit button for agent selection
agent_submitted = st.sidebar.button(
    "Apply Agent Selection",
    type="primary",
    use_container_width=True,
    key="apply_agents"
)

# Update the agent_type based on selection when submitted
if agent_submitted:
    if use_all_agents:
        st.session_state.agent_type = "All Agents"
    else:
        if not selected_agents:  # If no agents selected, default to all
            st.session_state.agent_type = "All Agents"
        elif len(selected_agents) == len(available_agents):  # If all agents selected
            st.session_state.agent_type = "All Agents"
        else:
            st.session_state.agent_type = selected_agents
    st.sidebar.success("Agent selection updated!")

# Add spacing in sidebar
st.sidebar.markdown("---")

# Initialize session state for navigation if not set
if 'current_page' not in st.session_state:
    st.session_state.current_page = "Home"

if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []

# Navigation Buttons - Updated styling
nav_col1, nav_col2, nav_col3 = st.sidebar.columns(3)

with nav_col1:
    home_btn = st.button(
        "Home",
        key="nav_Home",
        type="primary" if st.session_state.current_page == "Home" else "secondary",
        use_container_width=True,
        disabled=False  # Enable the button
    )

with nav_col2:
    report_btn = st.button(
        "Report",
        key="nav_Report",
        type="primary" if st.session_state.current_page == "Combined Report" else "secondary",
        use_container_width=True,
        disabled=False  # Enable the button
    )

with nav_col3:
    about_btn = st.button(
        "About",
        key="nav_About",
        type="primary" if st.session_state.current_page == "Combined Report" else "secondary",
        use_container_width=True,
        disabled=False  # Enable the button
    )

# Handle navigation button clicks
if home_btn:
    st.session_state.current_page = "Home"
    st.rerun()
elif report_btn:
    st.session_state.current_page = "Combined Report"
    st.rerun()
elif about_btn:
    st.session_state.current_page = "About"
    st.rerun()

# Current page
page = st.session_state.current_page

# ------------------------------------
# Custom CSS for NVIDIA-inspired Theme
# ------------------------------------
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
    color: #fff !important;               /* white text */
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
    background-color: #76B900 !important;
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
</style>


""", unsafe_allow_html=True)

# --------------------------------
# Home Page
# --------------------------------
if page == "Home":
    st.title("Welcome to the NVIDIA Multi-Agent Research Assistant")
    st.markdown("""
    This application integrates multiple agents to produce comprehensive research reports on NVIDIA:
    
    - **RAG Agent:** Retrieves historical quarterly reports from Pinecone (Year/Quarter).
    - **Web Search Agent:** Provides real-time insights via SerpAPI.
    - **Snowflake Agent:** Queries structured valuation metrics from Snowflake and displays charts.
    
    Use the navigation panel to generate a combined research report or learn more about the application.
    """)

# --------------------------------
# Combined Research Report
# --------------------------------
elif page == "Combined Report":
    st.title("NVIDIA Research Assistant")

    # Container for Chat History
    st.subheader("Research History")
    chat_container = st.container()
    with chat_container:
        st.write("Below are your previous queries and the assistant's responses.")
        st.write("You can scroll to see older messages.")
        st.markdown("""---""")
        # Display chat history
        if st.session_state.chat_history:
            for message in st.session_state.chat_history:
                if message["role"] == "user":
                    # Show user message
                    periods_text = "All Quarters" if message.get('search_type') == "All Quarters" else ", ".join(message.get('selected_periods', []))
                    st.markdown(f"""
                    <div class="user-message">
                        <div class="metadata">📅 {periods_text}</div>
                        <div>🔍 {message['content']}</div>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    # Show assistant message
                    st.markdown(f"""
                    <div class="assistant-message">
                        <div class="metadata">🤖 NVIDIA Research Assistant</div>
                        <div>{message["content"]}</div>
                    </div>
                    """, unsafe_allow_html=True)

    # ----------- Input form -----------
    st.markdown("---")
    with st.form(key="report_form", clear_on_submit=True):
        question = st.text_input(
            "Research Question",
            placeholder="What has driven NVIDIA's revenue growth in recent quarters?",
            key="question_input"
        )
        st_type = st.session_state.search_type
        selected_periods = (
            st.session_state.get("period_select", ["2023q1"])
            if st_type == "Specific Quarter"
            else ["all"]
        )
        submitted = st.form_submit_button("➤", use_container_width=True)

    # ----------- On Submit -----------
    if submitted and question:
        with st.spinner("🤖 Generating comprehensive NVIDIA analysis..."):
            agents_to_use = ["All Agents"] if use_all_agents else selected_agents
            payload = {
                "question": question,
                "search_type": st_type,
                "selected_periods": selected_periods,
                "agents": agents_to_use
            }
            try:
                response = requests.post(QUERY_URL, json=payload)
                if response.status_code == 200:
                    data = response.json()
                    st.session_state.chat_history.append({
                        "role": "user",
                        "content": question,
                        "search_type": st_type,
                        "selected_periods": selected_periods,
                        "agents": agents_to_use
                    })
                    if "Snowflake Agent" in agents_to_use:
                        content = data.get("valuation_data", {}).get("summary", "No Snowflake data available")
                    elif "RAG Agent" in agents_to_use:
                        content = data.get("rag_output", {}).get("result", "No RAG data available")
                    elif "Web Search Agent" in agents_to_use:
                        content = data.get("web_output", "No web search data available")
                    else:
                        content = data.get("final_report", "No report generated")
                    st.session_state.chat_history.append({
                        "role": "assistant",
                        "content": content,
                        "agents": agents_to_use,
                        "rag_output": data.get("rag_output", {}) if "RAG Agent" in agents_to_use else None,
                        "snowflake_data": data.get("valuation_data", {}) if "Snowflake Agent" in agents_to_use else None,
                        "web_output": data.get("web_output", {}) if "Web Search Agent" in agents_to_use else None
                    })
                    st.rerun()
                else:
                    st.error(f"❌ API Error: {response.status_code} - {response.text}")
            except Exception as e:
                st.error(f"❌ Error: {str(e)}")

    # ---------------------------- Detailed Results (Tabs) ----------------------------
    assistant_msgs = [msg for msg in st.session_state.chat_history if msg["role"] == "assistant"]
    if assistant_msgs:
        latest_assistant_msg = assistant_msgs[-1]
        final_report = latest_assistant_msg.get("content", "")
        rag_output = latest_assistant_msg.get("rag_output", {})
        snowflake_data = latest_assistant_msg.get("snowflake_data", {})
        st.markdown("---")
        st.subheader("Detailed Results")
        tabs_to_show = []
        if use_all_agents or len(selected_agents) == len(available_agents):
            tabs_to_show = ["Overview", "Sources & Web Results", "Financial Visualization"]
        else:
            if "Snowflake Agent" in selected_agents:
                tabs_to_show.append("Financial Visualization")
            if "RAG Agent" in selected_agents or "Web Search Agent" in selected_agents:
                tabs_to_show.append("Sources & Web Results")
            if not tabs_to_show:
                tabs_to_show = ["Overview"]
        tabs = st.tabs(tabs_to_show)
        if "Overview" in tabs_to_show:
            with tabs[tabs_to_show.index("Overview")]:
                st.markdown(latest_assistant_msg.get("content", ""), unsafe_allow_html=True)
        if "Sources & Web Results" in tabs_to_show:
            with tabs[tabs_to_show.index("Sources & Web Results")]:
                if st.session_state.agent_type in ["All Agents", "RAG Agent"]:
                    st.markdown("### Document Analysis")
                    if isinstance(rag_output, dict):
                        st.markdown(rag_output.get("result", "No RAG data available"))
                if st.session_state.agent_type in ["All Agents", "Web Search Agent"]:
                    st.markdown("### Web Search Results")
                    web_output = latest_assistant_msg.get("web_output", "No web search data available")
                    st.markdown(web_output)
        if "Financial Visualization" in tabs_to_show:
            with tabs[tabs_to_show.index("Financial Visualization")]:
                st.markdown("### NVIDIA Financial Metrics Visualization")
                
                # Display the stacked area chart generated by Snowflake Agent
                chart_file_path = "nvidia_stacked_area_chart.png"  # Ensure this path is correct
                try:
                    st.image(
                        chart_file_path,
                        caption="NVIDIA Metrics Over Time (Stacked Area Chart)",
                        use_column_width=True
                    )
                except FileNotFoundError:
                    st.error("❌ The visualization chart file could not be found. Please ensure it is generated correctly.")

                # Display additional metrics if available
                snowflake_data = latest_assistant_msg.get("snowflake_data", {})
                if snowflake_data:
                    if "metrics" in snowflake_data:
                        st.markdown("#### Key Metrics")
                        if isinstance(snowflake_data["metrics"], list):
                            st.dataframe(snowflake_data["metrics"])
                        else:
                            st.write(snowflake_data["metrics"])
                    else:
                        st.info("ℹ️ No additional metrics available from the Snowflake Agent.")
                else:
                    st.warning("⚠️ No data received from the Snowflake Agent.")

# --------------------------------
# About Page
# --------------------------------
elif page == "About":
    st.title("About NVIDIA Research Assistant")
    st.markdown("""
    **NVIDIA Multi-Agent Research Assistant** integrates:
    
    - **RAG Agent:** Uses Pinecone (index: `nvidia-reports`) with metadata filtering 
      (e.g., `2023q2`, `2024q1`) for historical quarterly reports.
    - **Web Search Agent:** Uses SerpAPI for real-time web search related to NVIDIA.
    - **Snowflake Agent:** Connects to Snowflake to query structured NVIDIA valuation measures and displays visual charts.
    
    **Usage Instructions:**
    - Go to the **Combined Report** page to generate a comprehensive research report.
    - Configure whether you want all quarters or specific quarters in the sidebar.
    - Enter your question at the bottom, then click the circular button to submit.
    
    **Developed by:** Team Name 4
    """)
