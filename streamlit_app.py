#!/usr/bin/env python
# coding: utf-8

# Set page config as the first Streamlit command
import streamlit as st
st.set_page_config(page_title="SWOT Analysis", page_icon="📊", layout="wide")

# importing other necessary libraries
import os
import google.generativeai as genai
import langchain
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.text_splitter import CharacterTextSplitter
from langchain.docstore.document import Document
from langchain.chains.summarize import load_summarize_chain
import tiktoken
import matplotlib.pyplot as plt
import re
import io
from pdfminer.high_level import extract_text
import numpy as np
from matplotlib.colors import LinearSegmentedColormap

# Application title and description
st.title("SWOT Analysis Application")
st.write("Upload a file (.txt or .pdf) or enter text below to generate SWOT Analysis:")

# displaying versions of libraries in the sidebar
st.sidebar.markdown("### Library Versions")
st.sidebar.markdown(f"google.generativeai: {genai.__version__}")
st.sidebar.markdown(f"streamlit: {st.__version__}")
st.sidebar.markdown(f"tiktoken: {tiktoken.__version__}")
st.sidebar.markdown(f"langchain: {langchain.__version__}")

# Initialize token counters in session state
if 'tokens_consumed' not in st.session_state:
    st.session_state.tokens_consumed = 0
if 'query_tokens' not in st.session_state:
    st.session_state.query_tokens = 0
if 'response_tokens' not in st.session_state:
    st.session_state.response_tokens = 0

# Get API key from Streamlit secrets
if 'GOOGLE_API_KEY' in st.secrets:
    api_key = st.secrets['GOOGLE_API_KEY']
    genai.configure(api_key=api_key)
else:
    st.error("API key not found in secrets. Please add GOOGLE_API_KEY to your Streamlit secrets.")
    st.stop()

# Example of using tiktoken for token counting
encoder = tiktoken.get_encoding("cl100k_base")

# Initialize the Gemini AI model
@st.cache_resource
def load_llm():
    return ChatGoogleGenerativeAI(
        model="gemini-1.5-pro-latest",
        temperature=0.7,
        max_tokens=8000
    )

llm = load_llm()

# Enhanced prompt template for more comprehensive insights
prompt_template = """
You are a world-class strategic business consultant at BCG with expertise in comprehensive company analysis.

Based on the following information about the company, please provide a detailed SWOT analysis:
{company_info}

When creating the SWOT analysis, consider the following aspects for each quadrant:

**Strengths:**
- Core competencies and unique value propositions
- Brand reputation and market position
- Financial health and performance metrics
- Intellectual property and proprietary technology
- Operational efficiency and quality control
- Talent pool and organizational culture
- Supply chain advantages and distribution networks

**Weaknesses:**
- Operational inefficiencies or bottlenecks
- Gaps in product/service offerings
- Financial constraints or concerns
- Talent or skill gaps
- Technology or infrastructure limitations
- Brand perception issues
- Geographic or market limitations

**Opportunities:**
- Emerging market trends and consumer behaviors
- Potential new market segments or geographies
- Technological innovations relevant to the industry
- Strategic partnership possibilities
- Regulatory changes that could be advantageous
- Competitor vulnerabilities
- Economic or demographic shifts

**Threats:**
- Competitive landscape intensification
- Disruptive technologies or business models
- Regulatory challenges or compliance issues
- Economic, political or environmental risks
- Supply chain vulnerabilities
- Changing consumer preferences or behaviors
- Potential talent drain or labor market challenges

For each point, include a brief explanation of why it's significant and, where possible, suggest potential strategic implications or actions.

Format your SWOT analysis as follows:
**Strengths:**
- [Strength 1]: [Brief explanation]
- [Strength 2]: [Brief explanation]
...

**Weaknesses:**
- [Weakness 1]: [Brief explanation]
- [Weakness 2]: [Brief explanation]
...

**Opportunities:**
- [Opportunity 1]: [Brief explanation]
- [Opportunity 2]: [Brief explanation]
...

**Threats:**
- [Threat 1]: [Brief explanation]
- [Threat 2]: [Brief explanation]
...

Please ensure that the analysis is comprehensive, insightful, and directly relevant to the company's specific situation.
"""

# Create a PromptTemplate instance
prompt = PromptTemplate(input_variables=["company_info"], template=prompt_template)

# Initialize the LLMChain with the prompt and LLM
llm_chain = LLMChain(prompt=prompt, llm=llm)

# Function for generating SWOT analysis
def get_swot_analysis(company_info: str):
    return llm_chain.run(company_info)

# Function to extract text from PDF files
def extract_text_from_pdf(pdf_file):
    pdf_bytes = pdf_file.read()
    text = extract_text(io.BytesIO(pdf_bytes))
    return text

# Helper functions for processing and displaying SWOT analysis
def convert_md_bold_to_html(text: str) -> str:
    """Converts double-asterisk Markdown to HTML bold tags."""
    return re.sub(r"\*\*(.*?)\*\*", r"<b>\1</b>", text)

def remove_single_asterisks(text: str) -> str:
    """Removes single asterisks followed by a space."""
    return re.sub(r"^\*\s+", "", text, flags=re.MULTILINE)

def parse_subheading_bullets(text: str):
    """Finds lines that begin with '* ' or '- ', then cleans and returns them."""
    lines = re.findall(r"^(?:\*|-)\s+(.*?)(?:\s*:\s*|\s*-\s*|\s*–\s*)(.*)", text, flags=re.MULTILINE)
    if not lines:
        # Try simpler pattern if the more complex one fails
        lines = re.findall(r"^(?:\*|-)\s+(.*)", text, flags=re.MULTILINE)
        bullet_points = []
        for line in lines:
            line = convert_md_bold_to_html(line)
            line = remove_single_asterisks(line)
            bullet_points.append(line.strip())
        return bullet_points
    else:
        # Process lines with explanations
        bullet_points = []
        for point, explanation in lines:
            line = f"{point.strip()}: {explanation.strip()}"
            line = convert_md_bold_to_html(line)
            line = remove_single_asterisks(line)
            bullet_points.append(line.strip())
        return bullet_points

# HTML display function for SWOT quadrants
def display_swot_analysis_html(strengths, weaknesses, opportunities, threats):
    def render_quadrant(content: str, title: str, color: str):
        st.markdown(
            f"""
            <div style="background-color:{color}; padding: 10px; border-radius: 10px;">
                <h3 style="color:white;">{title}</h3>
                <ul style="color:white;">
            """,
            unsafe_allow_html=True
        )

        bullet_lines = parse_subheading_bullets(content)
        if not bullet_lines:
            bullet_lines = [content.strip()]

        for line in bullet_lines:
            st.markdown(f"<li>{line}</li>", unsafe_allow_html=True)

        st.markdown("</ul></div>", unsafe_allow_html=True)

    col1, col2 = st.columns(2)
    with col1:
        render_quadrant(strengths, "Strengths", "#4CAF50")
    with col2:
        render_quadrant(weaknesses, "Weaknesses", "#F44336")

    col3, col4 = st.columns(2)
    with col3:
        render_quadrant(opportunities, "Opportunities", "#2196F3")
    with col4:
        render_quadrant(threats, "Threats", "#FF9800")

# Matplotlib visualization for SWOT quadrants
def plot_swot_quadrants(strengths, weaknesses, opportunities, threats):
    # Create figure and axis
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Remove axis ticks and spines
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)
    
    # Split the plot into quadrants
    ax.axhline(y=0.5, color='black', linestyle='-', alpha=0.7)
    ax.axvline(x=0.5, color='black', linestyle='-', alpha=0.7)
    
    # Define colors for each quadrant
    colors = ['#4CAF50', '#F44336', '#2196F3', '#FF9800']
    
    # Parse bullet points for each quadrant
    strength_points = parse_subheading_bullets(strengths) or [strengths.strip()]
    weakness_points = parse_subheading_bullets(weaknesses) or [weaknesses.strip()]
    opportunity_points = parse_subheading_bullets(opportunities) or [opportunities.strip()]
    threat_points = parse_subheading_bullets(threats) or [threats.strip()]
    
    # Limit to 5 points per quadrant for readability
    max_points = 5
    
    # Add quadrant titles and bullet points
    # Strengths (top-left)
    ax.text(0.25, 0.95, 'STRENGTHS', horizontalalignment='center', 
            fontsize=12, fontweight='bold', color='white',
            bbox=dict(facecolor=colors[0], alpha=0.8, boxstyle='round,pad=0.5'))
    
    for i, point in enumerate(strength_points[:max_points]):
        ax.text(0.05, 0.85 - i*0.06, f"• {point[:60]}{'...' if len(point) > 60 else ''}", 
                fontsize=8, wrap=True)
    
    # Weaknesses (top-right)
    ax.text(0.75, 0.95, 'WEAKNESSES', horizontalalignment='center', 
            fontsize=12, fontweight='bold', color='white',
            bbox=dict(facecolor=colors[1], alpha=0.8, boxstyle='round,pad=0.5'))
    
    for i, point in enumerate(weakness_points[:max_points]):
        ax.text(0.55, 0.85 - i*0.06, f"• {point[:60]}{'...' if len(point) > 60 else ''}", 
                fontsize=8, wrap=True)
    
    # Opportunities (bottom-left)
    ax.text(0.25, 0.45, 'OPPORTUNITIES', horizontalalignment='center', 
            fontsize=12, fontweight='bold', color='white',
            bbox=dict(facecolor=colors[2], alpha=0.8, boxstyle='round,pad=0.5'))
    
    for i, point in enumerate(opportunity_points[:max_points]):
        ax.text(0.05, 0.35 - i*0.06, f"• {point[:60]}{'...' if len(point) > 60 else ''}", 
                fontsize=8, wrap=True)
    
    # Threats (bottom-right)
    ax.text(0.75, 0.45, 'THREATS', horizontalalignment='center', 
            fontsize=12, fontweight='bold', color='white',
            bbox=dict(facecolor=colors[3], alpha=0.8, boxstyle='round,pad=0.5'))
    
    for i, point in enumerate(threat_points[:max_points]):
        ax.text(0.55, 0.35 - i*0.06, f"• {point[:60]}{'...' if len(point) > 60 else ''}", 
                fontsize=8, wrap=True)
    
    # Add light color backgrounds to each quadrant
    rect1 = plt.Rectangle((0, 0.5), 0.5, 0.5, color=colors[0], alpha=0.1)
    rect2 = plt.Rectangle((0.5, 0.5), 0.5, 0.5, color=colors[1], alpha=0.1)
    rect3 = plt.Rectangle((0, 0), 0.5, 0.5, color=colors[2], alpha=0.1)
    rect4 = plt.Rectangle((0.5, 0), 0.5, 0.5, color=colors[3], alpha=0.1)
    
    ax.add_patch(rect1)
    ax.add_patch(rect2)
    ax.add_patch(rect3)
    ax.add_patch(rect4)
    
    # Set the title
    plt.title("SWOT Analysis", fontsize=16, fontweight='bold', pad=20)
    
    # Set axis limits
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    
    # Show plot in Streamlit
    st.pyplot(fig)

# Input options: File upload or text input
file_type = st.radio("Choose input method:", ["Upload File", "Enter Text"])

text = None

if file_type == "Upload File":
    uploaded_file = st.file_uploader("Choose a file", type=["txt", "pdf"])
    
    if uploaded_file is not None:
        # Process based on file type
        if uploaded_file.type == "application/pdf":
            with st.spinner("Extracting text from PDF..."):
                text = extract_text_from_pdf(uploaded_file)
                if text:
                    st.success(f"PDF processed successfully. Extracted {len(text)} characters.")
                    with st.expander("Preview extracted text"):
                        st.text(text[:1000] + ("..." if len(text) > 1000 else ""))
                else:
                    st.error("Failed to extract text from the PDF.")
        else:  # txt file
            text = uploaded_file.read().decode("utf-8")
            with st.expander("Preview uploaded text"):
                st.text(text[:1000] + ("..." if len(text) > 1000 else ""))
else:
    text_input = st.text_area("Enter company information:")
    if text_input:
        text = text_input

# Add a button to generate the SWOT analysis
if st.button("Generate SWOT Analysis"):
    if text:
        # Generate and display the SWOT Analysis
        with st.spinner('Generating SWOT Analysis... This may take a minute.'):
            swot_output = get_swot_analysis(text)
        
        # Count tokens
        query_tokens = len(encoder.encode(text))
        response_tokens = len(encoder.encode(swot_output))
        
        # Update token counts in session state
        st.session_state.query_tokens += query_tokens
        st.session_state.response_tokens += response_tokens
        st.session_state.tokens_consumed += (query_tokens + response_tokens)
        
        # Parse the SWOT output
        sections = ["Strengths", "Weaknesses", "Opportunities", "Threats"]
        swot_blocks = {s: "" for s in sections}
    
        # Regex pattern to match each section until the next section or end of string
        for section in sections:
            pattern = rf"\*\*{section}:\*\*\s*((?:(?!\*\*(?:Strengths|Weaknesses|Opportunities|Threats):\*\*).)*)"
            match = re.search(pattern, swot_output, re.DOTALL)
            if match:
                swot_blocks[section] = match.group(1).strip()
            else:
                swot_blocks[section] = ""
        
        # Display results with tabs for different visualizations
        st.subheader("SWOT Analysis Results")
        tabs = st.tabs(["HTML Visualization", "Matplotlib Visualization", "Raw Output"])
        
        with tabs[0]:
            # Display the SWOT quadrants using HTML
            display_swot_analysis_html(
                swot_blocks["Strengths"],
                swot_blocks["Weaknesses"],
                swot_blocks["Opportunities"],
                swot_blocks["Threats"]
            )
        
        with tabs[1]:
            # Display the SWOT quadrants using Matplotlib
            plot_swot_quadrants(
                swot_blocks["Strengths"],
                swot_blocks["Weaknesses"],
                swot_blocks["Opportunities"],
                swot_blocks["Threats"]
            )
        
        with tabs[2]:
            # Show raw output
            st.markdown(swot_output)
            
    else:
        st.info("Please upload a file or enter text to generate the SWOT analysis.")

# Display token usage in sidebar
st.sidebar.markdown("### Token Usage")
st.sidebar.markdown(f"Total Tokens Consumed: {st.session_state.tokens_consumed}")
st.sidebar.markdown(f"Query Tokens: {st.session_state.query_tokens}")
st.sidebar.markdown(f"Response Tokens: {st.session_state.response_tokens}")

# Reset token counts button
if st.sidebar.button("Reset Token Counters"):
    st.session_state.tokens_consumed = 0
    st.session_state.query_tokens = 0
    st.session_state.response_tokens = 0
    st.sidebar.success("Token counters reset.")
