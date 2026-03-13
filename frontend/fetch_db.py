
# Core AI step: Get embedding for the search query using Azure OpenAI
import datetime
import os
from venv import logger
import weaviate
from weaviate.classes.query import Filter
import requests
from datetime import datetime, timedelta, timezone


# Connect to Weaviate
# @st.cache_resource
# def get_client():
#     return weaviate.connect_to_local()

client = weaviate.connect_to_local()

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



def build_my_where_clause(llm_extracted_json, allowed_fields=None):
    """
    Convert LLM extracted JSON metadata to Weaviate 'where' clause.
    Skips any null values.
    Handles special types like date or arrays.
    """
    semantic_query = llm_extracted_json.pop("semantic_query", None)

    filters = []

    for key, value in llm_extracted_json.items():

        # Skip semantic query
        if key == "semantic_query":
            continue

        # 🚀 Skip None / empty values
        if value in [None, "None", "NULL", "", [], {}]:
            continue

        # Skip hallucinated fields
        if allowed_fields and key not in allowed_fields:
            continue

        if key.endswith("_timestamp") and isinstance(value, dict):
            start_str = value.get("start")
            end_str = value.get("end")

            if start_str and start_str != "None":
                start_dt = datetime.fromisoformat(start_str)
                filters.append(Filter.by_property(key).greater_or_equal(start_dt))

            if end_str and end_str != "None":
                end_dt = datetime.fromisoformat(end_str)
                filters.append(Filter.by_property(key).less_or_equal(end_dt))

        if key.endswith("_number"):
            #st.write({int(value)})
            # st.write({type(value)})
            filters.append(Filter.by_property(key).equal(int(value)))

        elif isinstance(value, str):
            filters.append(Filter.by_property(key).equal(value))


    where_filter = Filter.all_of(filters) if filters else None

    return where_filter, semantic_query


def get_db_collection(semantic_query, where_filter, query_vector):
    try:
        print("Where cfilter = ", where_filter)
        collection = client.collections.get("Metadata")


        response = collection.query.hybrid(
            query=semantic_query,
            vector=query_vector,
            alpha=0.5,
            limit=10,
            filters=where_filter
        )

        # Extract results
        metadata = response.objects if response.objects else []

        #metadata = [obj.properties for obj in response.objects]

        if not metadata:
            #logger.info("No results found.")
            print("No results found")

        return metadata
    except:
        return []
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

    # except Exception as e:
    #     st.error(f"Error during search: {e}")
