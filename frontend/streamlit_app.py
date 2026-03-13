import streamlit as st
import weaviate
import os
import requests
from pathlib import Path
from dotenv import load_dotenv
from openai import AzureOpenAI
import sys
import os
from filter_extractor import extract_filters
from fetch_db import build_my_where_clause, get_db_collection, get_query_embedding
from display_results import present_results, fetch

# Add parent directory to Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


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

# Always load the workspace-level .env regardless of launch directory.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
load_dotenv(PROJECT_ROOT / ".env")

st.set_page_config(
    page_title="AI Confluence Search",
    page_icon="🔎",
    layout="wide"
)

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

# Set the background
set_background(r"C:\Users\TSAN01\PycharmProjects\labweek_2026\frontend\background.png")
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

# Connect to Weaviate
@st.cache_resource
def get_client():
    return weaviate.connect_to_local()

client = get_client()

# Core AI step: Get embedding for the search query using Azure OpenAI
def get_query_embedding(text):
    api_key = os.getenv("AZURE_OPENAI_KEY")
    endpoint = os.getenv("AZURE_OPENAI_ENDPOINT") 
    deployment = os.getenv("AZURE_OPENAI_EMBEDDING_DEPLOYMENT")
    api_version = os.getenv("AZURE_OPENAI_API_VERSION")
    
    if not api_key or not endpoint:
        raise RuntimeError(
            "Set AZURE_OPENAI_KEY and AZURE_OPENAI_ENDPOINT in your .env file."
        )

    url = (
        f"{endpoint.rstrip('/')}"
        f"/openai/deployments/{deployment}/embeddings?api-version={api_version}"
    )
    response = requests.post(
        url,
        headers={
            "api-key": api_key,
            "Content-Type": "application/json",
        },
        json={"input": text},
        timeout=30,
    )
    response.raise_for_status()
    return response.json()["data"][0]["embedding"]

# Text input for the search query
query = st.text_input("Search internal knowledge")

if st.button("Search") and query:

    with st.spinner("Optimizing query..."):
        llm_extract = extract_filters(query)
        #st.write(llm_extract)
        where_filter, semantic_query = build_my_where_clause(llm_extract, allowed_fields=ALLOWED_FILTER_FIELDS)
        st.write(f"**Semantic Query:** {semantic_query}")
        query_vector = get_query_embedding(
            "Convert this query into an embedding for semantic search of Confluence knowledge:" + (semantic_query  or "")
        )
        #st.write(where_filter)
        results = get_db_collection(semantic_query, where_filter, query_vector)
        #st.write(results)
        enriched = present_results(results, query)
        # enriched = fetch(
        #     query
        # )
        #rewritten_query = rewrite_query(query)
        #st.write(f"**Rewritten Query:** {enriched}")
        #enriched = fetch(query)

        # Case 1: LLM returned valid JSON (normal mode)
        if isinstance(enriched, dict) and "results" in enriched:
            st.subheader("Results Summary")
            st.write(enriched["summary"])

            for item in enriched["results"]:
                st.markdown(f"### {item['title']}")
                st.markdown(f"**Section:** {item['section']}")
                st.markdown(f"**Updated:** {item['updated']}  |  **Created:** {item['created']}")
                st.markdown(f"**Author:** {item['author']}  |  **Created by:** {item['created_by']}")
                st.write(item["snippet"])
                st.caption(item["relevance_reason"])

                if item.get("is_outdated"):
                    st.error("⚠️ This page is older than 2 years and may need cleanup.")
                    if item.get("contact_author"):
                        st.markdown(f"[Email author]({item['contact_author']})")

                st.markdown("---")


        # Case 2: JSON failed → show human-readable Markdown from LLM
        else:
            #st.error("⚠️ The AI returned human-readable text instead of JSON. Showing fallback view.")

            raw = enriched.get("raw_response", "")

            # Pretty print Markdown
            st.markdown(raw)
        # query_vector = get_query_embedding(
        #     "Convert this query into an embedding for semantic search of Confluence knowledge:" + rewritten_query
        # )
        #
        # # Show the vector
        # #st.write("First 10 dimensions of embedding:", query_vector[:10])
        #
        # # Perform semantic search in the 'Metadata' class
        # try:
        #
        #     collection = client.collections.get("Metadata")
        #     response = collection.query.near_vector(
        #         near_vector=query_vector,
        #         limit=10,
        #         return_properties=["lastmodified_timestamp", "space_name", "title", "url"]
        #     )
        #
        #     # Extract results
        #     metadata = response.objects if response.objects else []
        #
        #     if not metadata:
        #         st.info("No results found.")
        #     else:
        #         for idx, obj in enumerate(metadata, start=1):
        #             props = obj.properties
        #             lastmodified_timestamp = props.get("lastmodified_timestamp", "Unknown Date")
        #             title = props.get("title", "No Title")
        #             if props.get('url'):
        #                 base_url = "https://www.stb.bskyb.com/confluence"
        #                 url = base_url + props.get("url", "")
        #                 st.markdown(f"### {title} 🔗 ({url})")
        #             else:
        #                 st.markdown(f"### {title}")
        #             st.markdown(f"**Last Modified:** {lastmodified_timestamp}")
        #             st.markdown("---")
        #
        # except Exception as e:
        #     st.error(f"Error during search: {e}")

