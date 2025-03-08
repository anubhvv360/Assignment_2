# Assignment 2

A simple Streamlit app showing the GDP of different countries in the world.

[![Open in Streamlit](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://swotanalysis.streamlit.app/)

# SWOT Analysis Application

This is a Streamlit application that generates SWOT (Strengths, Weaknesses, Opportunities, Threats) analyses for companies using Google's Gemini AI model.

## Features

- Upload text files or enter text directly to analyze a company
- Beautiful visualization of SWOT analysis in four quadrants
- Token usage tracking
- Responsive design

## Setup

### Local Development

1. Clone this repository
2. Install dependencies:
   ```
   pip install -r requirements.txt
   ```
3. Create a `.streamlit/secrets.toml` file with your Gemini API key:
   ```toml
   GOOGLE_API_KEY = "your_api_key_here"
   ```
4. Run the application:
   ```
   streamlit run app.py
   ```

### Deployment

This application is designed to be deployed on Streamlit Cloud:

1. Push your repository to GitHub (without secrets.toml)
2. Connect your repository to Streamlit Cloud
3. Add your Gemini API key in the Streamlit Cloud secrets management interface

## How It Works

The application uses Google's Gemini AI model through the LangChain framework to generate a structured SWOT analysis based on input text. The results are displayed in a visually appealing four-quadrant layout.

## Requirements

See `requirements.txt` for a full list of dependencies.
