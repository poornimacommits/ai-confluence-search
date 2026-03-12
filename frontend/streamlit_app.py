from datetime import datetime, timedelta, timezone
import json

from filter_extractor import extract_filters
import streamlit as st
import weaviate
import os
import requests
from pathlib import Path
from dotenv import load_dotenv
from openai import AzureOpenAI
from weaviate.classes.query import Filter
from weaviate.collections.classes.filters import _Filters

from fetch_db import build_my_where_clause, get_db_collection, get_query_embedding

# Always load the workspace-level .env regardless of launch directory.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
load_dotenv(PROJECT_ROOT / ".env")

# Apply background image with CSS
def set_background(png_file):
    with open(png_file, "rb") as f:
        data = f.read()
    import base64
    encoded = base64.b64encode(data).decode()
    st.markdown(
        f"""
        <style>
        .stApp {{
            background-image: url("data:image/png;base64,{encoded}");
            background-size: cover;
            background-position: center;
            background-attachment: fixed;
        }}
        /* Optional: Make text boxes and containers more readable */
        .stTextInput, .stButton {{
            background-color: rgba(255, 255, 255, 0.85);
            border-radius: 10px;
        }}
        </style>
        """,
        unsafe_allow_html=True
    )



st.set_page_config(
    page_title="AI Confluence Search",
    page_icon="🔎",
    layout="wide"
)
# Set the background
set_background("background.png")
st.title("🔎 AI-Powered Confluence Search")

st.markdown("Find the **most reliable internal documentation instantly**.")

# Initialize GPT-4 client for query rewriting
client_gpt = AzureOpenAI(
    api_key=os.getenv("AZURE_OPENAI_KEY"),
    api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT")
)

def rewrite_query(user_query):
    prompt = f"""
    Rewrite the user query so it is clear and optimized for semantic search 
    in a knowledge base containing technical documentation and Confluence pages.

    User query:
    {user_query}

    Rewritten query:
    """
    response = client_gpt.chat.completions.create(
        model=os.getenv("AZURE_OPENAI_QUERY_REWRITE_DEPLOYMENT"),
        temperature=0,
        messages=[
            {"role": "system", "content": "You improve search queries."},
            {"role": "user", "content": prompt}
        ]
    )
    return response.choices[0].message.content.strip()

ALLOWED_FILTER_FIELDS = {
    "lastmodified_author",
    "space_name",
    "lastmodified_timestamp",
    "created_by",
    "created_timestamp",
    "page_id",
    "version_number",
    "space_key",
    "ancestor_titles"
}



# Text input for the search query
query = st.text_input("Search internal knowledge")

if st.button("Search") and query:

    with st.spinner("Preparing hybrid query..."):

        llm_ext = extract_filters(query)

        #llm_ext = json.loads(filters, indent=2)

        st.write(llm_ext)

        where_filter, semantic_query = build_my_where_clause(llm_ext,
                allowed_fields=ALLOWED_FILTER_FIELDS)

        st.write(f"**Semantic Query:** {semantic_query}")
        query_vector = get_query_embedding(
            "Convert this query into an embedding for semantic search of Confluence knowledge:" + semantic_query
        )
        
        get_db_collection(semantic_query, where_filter, query_vector)
        
        #semantic_query = llm_extracted_json.get("semantic_query")
        

        # Show the vector
        #st.write("First 10 dimensions of embedding:", query_vector[:10])


