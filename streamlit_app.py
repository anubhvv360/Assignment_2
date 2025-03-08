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

# 1. Application title and description
st.title("SWOT Analysis Application")
st.write("Upload a file (.txt or .pdf) or enter text below to generate SWOT Analysis:")

# 2. Displaying versions of libraries in the sidebar
st.sidebar.markdown("### Library Versions")
st.sidebar.markdown(f"google.generativeai: {genai.__version__}")
st.sidebar.markdown(f"streamlit: {st.__version__}")
st.sidebar.markdown(f"tiktoken: {tiktoken.__version__}")
st.sidebar.markdown(f"langchain: {langchain.__version__}")

# 3. Initialize token counters in session state
if 'tokens_consumed' not in st.session_state:
    st.session_state.tokens_consumed = 0
if 'query_tokens' not in st.session_state:
    st.session_state.query_tokens = 0
if 'response_tokens' not in st.session_state:
    st.session_state.response_tokens = 0

# 4. Get API key from Streamlit secrets
if 'GOOGLE_API_KEY' in st.secrets:
    api_key = st.secrets['GOOGLE_API_KEY']
    genai.configure(api_key=api_key)
else:
    st.error("API key not found in secrets. Please add GOOGLE_API_KEY to your Streamlit secrets.")
    st.stop()

# 5. Example of using tiktoken for token counting
encoder = tiktoken.get_encoding("cl100k_base")

# 6. Initialize the Gemini AI model
@st.cache_resource
def load_llm():
    return ChatGoogleGenerativeAI(
        model="gemini-1.5-pro-latest",
        temperature=0.7,
        max_tokens=8000
    )

llm = load_llm()

# 7. Enhanced prompt template for more comprehensive insights
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

# 8. Initialize the LLMChain with the prompt and LLM
llm_chain = LLMChain(prompt=prompt, llm=llm)

# Function for generating SWOT analysis
def get_swot_analysis(company_info: str):
    return llm_chain.run(company_info)

# 9. Function to extract text from PDF files
def extract_text_from_pdf(pdf_file):
    pdf_bytes = pdf_file.read()
    text = extract_text(io.BytesIO(pdf_bytes))
    return text

# 10. Helper functions for processing and displaying SWOT analysis
def convert_md_bold_to_html(text: str) -> str:
    """Converts double-asterisk Markdown to HTML bold tags."""
    return re.sub(r"\*\*(.*?)\*\*", r"<b>\1</b>", text)

def remove_single_asterisks(text: str) -> str:
    """Removes single asterisks followed by a space."""
    return re.sub(r"^\*\s+", "", text, flags=re.MULTILINE)

def parse_subheading_bullets(text: str):
    """
    Finds lines that begin with '* ' or '- ', then cleans and returns them.
    Also handles lines in the format: '* Something: Explanation'
    """
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

def display_swot_analysis_html(strengths, weaknesses, opportunities, threats):
    """HTML-based display for the SWOT quadrants."""

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


# 11. NEW Matplotlib visualization function
import textwrap

def plot_swot_quadrants(strengths, weaknesses, opportunities, threats):
    """
    Creates a 2x2 subplot for Strengths, Weaknesses, Opportunities, and Threats.
    Summarizes bullet points (truncates lines, removes HTML tags/headings).
    """

    def clean_and_summarize(lines, max_points=5, max_line_length=100):
        """
        1. Remove HTML tags (<b> etc.)
        2. Remove references to headings like **Strengths:** if any
        3. Limit bullet points to `max_points`
        4. Truncate each line to `max_line_length` characters
        """
        cleaned = []
        # Patterns to remove headings like **Strengths:**
        heading_patterns = [r"\*\*Strengths:\*\*", r"\*\*Weaknesses:\*\*",
                            r"\*\*Opportunities:\*\*", r"\*\*Threats:\*\*"]
        for i, line in enumerate(lines):
            # Remove HTML tags
            line = re.sub(r"</?b>", "", line)
            line = re.sub(r"<.*?>", "", line)

            # Remove heading patterns
            for pattern in heading_patterns:
                line = re.sub(pattern, "", line, flags=re.IGNORECASE)

            # Truncate if too long
            if len(line) > max_line_length:
                line = line[:max_line_length].rstrip() + "..."

            cleaned.append(line.strip())

        # Limit bullet points
        return cleaned[:max_points]

    # Parse bullet points or fallback to raw text
    strength_points = parse_subheading_bullets(strengths) or [strengths.strip()]
    weakness_points = parse_subheading_bullets(weaknesses) or [weaknesses.strip()]
    opportunity_points = parse_subheading_bullets(opportunities) or [opportunities.strip()]
    threat_points = parse_subheading_bullets(threats) or [threats.strip()]

    # Clean and summarize
    strength_points = clean_and_summarize(strength_points)
    weakness_points = clean_and_summarize(weakness_points)
    opportunity_points = clean_and_summarize(opportunity_points)
    threat_points = clean_and_summarize(threat_points)

    # Combine each set of bullet points into a single string
    strength_text = "\n".join([f"• {pt}" for pt in strength_points])
    weakness_text = "\n".join([f"• {pt}" for pt in weakness_points])
    opportunity_text = "\n".join([f"• {pt}" for pt in opportunity_points])
    threat_text = "\n".join([f"• {pt}" for pt in threat_points])

    # Create 2x2 subplots
    fig, axs = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle("SWOT Analysis", fontsize=16, fontweight='bold', y=0.98)

    # Top-left: Strengths
    axs[0, 0].set_title("Strengths", color="#4CAF50", fontsize=14, fontweight='bold')
    axs[0, 0].text(
        0.5, 0.5,
        strength_text,
        ha='center',
        va='center',
        fontsize=10,
        wrap=True
    )
    axs[0, 0].set_axis_off()

    # Top-right: Weaknesses
    axs[0, 1].set_title("Weaknesses", color="#F44336", fontsize=14, fontweight='bold')
    axs[0, 1].text(
        0.5, 0.5,
        weakness_text,
        ha='center',
        va='center',
        fontsize=10,
        wrap=True
    )
    axs[0, 1].set_axis_off()

    # Bottom-left: Opportunities
    axs[1, 0].set_title("Opportunities", color="#2196F3", fontsize=14, fontweight='bold')
    axs[1, 0].text(
        0.5, 0.5,
        opportunity_text,
        ha='center',
        va='center',
        fontsize=10,
        wrap=True
    )
    axs[1, 0].set_axis_off()

    # Bottom-right: Threats
    axs[1, 1].set_title("Threats", color="#FF9800", fontsize=14, fontweight='bold')
    axs[1, 1].text(
        0.5, 0.5,
        threat_text,
        ha='center',
        va='center',
        fontsize=10,
        wrap=True
    )
    axs[1, 1].set_axis_off()

    # Show plot in Streamlit
    st.pyplot(fig)


# 12. Input options: File upload or text input
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

# 13. Add a button to generate the SWOT analysis
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

# 14. Display token usage in sidebar
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
