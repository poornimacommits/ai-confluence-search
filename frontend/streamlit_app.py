import streamlit as st
import requests

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

        response = requests.get(
            "http://localhost:8000/api/search",
            params={"query": query}
        )

        data = response.json()

        st.success(f"Refined Query: {data['refined_query']}")

        for r in data["results"]:

            score = r["score"]

            color = "green"
            if score < 7:
                color = "orange"
            if score < 4:
                color = "red"

            st.markdown("---")

            st.markdown(
                f"""
### [{r['title']}]({r['url']})

**Reliability Score:** :{color}[{score}/10]

{r['excerpt']}
"""
            )