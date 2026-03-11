import streamlit as st
import weaviate
import os
import requests
from pathlib import Path
from dotenv import load_dotenv


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


def resolve_background_path(file_name):
    app_dir = Path(__file__).resolve().parent
    local_path = app_dir / file_name
    if local_path.exists():
        return local_path
    cwd_path = Path.cwd() / file_name
    if cwd_path.exists():
        return cwd_path
    raise FileNotFoundError(f"Background image not found: {local_path} or {cwd_path}")


# Set the background
set_background(resolve_background_path("background.png"))
st.title("🔎 AI-Powered Confluence Search")

st.markdown("Find the **most reliable internal documentation instantly**.")

# Connect to Weaviate
@st.cache_resource
def get_client():
    return weaviate.connect_to_local()

client = get_client()

# Always load the workspace-level .env regardless of launch directory.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
load_dotenv(PROJECT_ROOT / ".env")

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

    with st.spinner("Searching..."):
        query_vector = get_query_embedding(
            "Convert this query into an embedding for semantic search of Confluence knowledge:" + query
        )

        # Show the vector
        #st.write("First 10 dimensions of embedding:", query_vector[:10])

        # Perform semantic search in the 'Metadata' class
        try:
            collection = client.collections.get("Metadata")
            response = collection.query.near_vector(
                near_vector=query_vector,
                limit=10,
                return_properties=["last_modified", "space_name", "title", "url"]
            )
        
            # Extract results
            metadata = response.objects if response.objects else []
        
            if not metadata:
                st.info("No results found.")
            else:
                for idx, obj in enumerate(metadata, start=1):
                    props = obj.properties
                    last_modified = props.get("last_modified", "Unknown Date")
                    title = props.get("title", "No Title")
                    if props.get('url'):
                        base_url = "https://www.stb.bskyb.com/confluence"
                        url = base_url + props.get("url", "")
                        st.markdown(f"### {title} 🔗 ({url})")
                    else:
                        st.markdown(f"### {title}")
                    st.markdown(f"**Last Modified:** {last_modified}")
                    st.markdown("---")

        except Exception as e:
            st.error(f"Error during search: {e}")

