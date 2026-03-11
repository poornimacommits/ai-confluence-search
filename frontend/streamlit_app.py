import streamlit as st
import weaviate
from sentence_transformers import SentenceTransformer


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
set_background("C:\\Users\\POMY1994\\VSProjects\\ai-confluence-search\\frontend\\background.png")
st.title("🔎 AI-Powered Confluence Search")

st.markdown("Find the **most reliable internal documentation instantly**.")

# Connect to Weaviate
client = weaviate.connect_to_local()

@st.cache_resource
def load_model():
    return SentenceTransformer("BAAI/bge-small-en-v1.5")

model = load_model()

# Text input for the search query
query = st.text_input("Search internal knowledge")

if st.button("Search") and query:

    with st.spinner("Searching..."):
        query_vector = model.encode("Convert this query into an embedding for semantic search of Confluence knowledge:" + query
        ).tolist()

        # Show the vector
        #st.write("First 10 dimensions of embedding:", query_vector[:10])

        # Perform semantic search in the 'Metadata' class
        try:
            collection = client.collections.get("Metadata")
            #response = collection.query.near_text(
                #query=query,
                #limit=5,
                #return_properties=["title", "url", "space", "status", "lastModified"]
            #)
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
                    space = props.get("space_name", "Unknown Space")
                    title = props.get("title", "No Title")
                    if props.get('url'):
                        base_url = "https://www.stb.bskyb.com/confluence"
                        url_suffix = props.get('url')
                        url = f"{base_url}{url_suffix}"
                        st.markdown(f"### {title} 🔗 ({url})")
                    else:
                        st.markdown(f"### {title}")
                    st.markdown(f"**Last Modified:** {last_modified}")
                    st.markdown("---")

        except Exception as e:
            st.error(f"Error during search: {e}")

