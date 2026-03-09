import streamlit as st
import requests
import os
import weaviate

# Connect to Weaviate
client = weaviate.connect_to_local()

st.set_page_config(
    page_title="AI Confluence Search",
    page_icon="🔎",
    layout="wide"
)

st.title("🔎 AI-Powered Confluence Search")

st.markdown("Find the **most reliable internal documentation instantly**.")

query = st.text_input("Search internal knowledge")

if st.button("Search"):

    with st.spinner("Searching..."):

        # Get the Article collection
        articles = client.collections.get("Article")

        # Fetch all objects
        all_objects = articles.query.fetch_objects(limit=100)

        st.write({len(all_objects.objects)})
        
        # Display each object
        for obj in all_objects.objects:
            st.write(f"UUID: {obj.uuid}")
            st.write(f"Content: {obj.properties['content']}")
            st.write(f"Vector: {obj.vector}")
            st.write("-" * 50)
       
client.close()  

