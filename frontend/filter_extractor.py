from dotenv import load_dotenv
import os
from openai import OpenAI, AzureOpenAI
from datetime import datetime, timedelta
import json
import re

today = datetime.utcnow().date().isoformat()

load_dotenv()
API_KEY = os.getenv("AZURE_OPENAI_KEY")  # Ensure your .env has

# ---- Set your OpenAI API key ----
endpoint = "https://ai-confluence-clean-up.openai.azure.com/"
model_name = "text-embedding-3-small"
deployment = "text-embedding-3-small"

api_version = "2024-02-01-preview"

gpt_model_name = "gpt-4o"
gpt_deployment = "gpt-4o"

subscription_key = "<your-api-key>"
api_version_gpt = "2024-12-01-preview"

# client = AzureOpenAI(
#     api_version=api_version_gpt,
#     azure_endpoint=endpoint,
#     api_key=API_KEY
# )
#
# def embed_text(text: str):
#     response = client.embeddings.create(
#         input=text,
#         model=model_name
#     )
#     return response.data[0].embedding

# -------------------------
# GPT-4 Hybrid Query
# -------------------------
# We'll use GPT-4 to extract metadata filters from the user query
def parse_gpt_json(response_text) -> dict:
    # Remove markdown code fences if present
    cleaned = re.sub(r"^```json\s*|\s*```$", "", response_text.strip(), flags=re.MULTILINE)
    # Remove any zero-width spaces or non-breaking spaces
    cleaned = cleaned.replace("\u200b", "").replace("\xa0", " ").strip()
    return json.loads(cleaned)


# ---------------------------
# Extract filters using GPT
# ---------------------------
def extract_filters(query: str):
    """
    Returns a dictionary of metadata filters extracted from natural language query.
    Example output: {"author": "Alice"}
    """
    today = datetime.utcnow().date()
    prompt_1 = f"""
    You are a helpful assistant and you are in today's date: {today}. 
    Your task is to extract metadata filters from a user's natural language query about Confluence pages.

    Extract the metadata filters from the query based on the available fields:
    Available metadata fields:
    - lastmodified_author: the person who last modified the page (may have multiple words)
    - space_name: workspace name (may have multiple words)
    - lastmodified_timestamp: date when page was last modified (YYYY-MM-DD)
    - created_by: the person who originally created the page (may have multiple words)
    - created_timestamp: date when page was originally created (YYYY-MM-DD)
    - page_id: unique identifier of the page (numeric)
    - version_number: version of the page (numeric)
    - space_key: short code for space (alphanumeric)
    - ancestor_titles: list of parent page titles (may have multiple words)


    Task: Extract metadata filters from the user's query.
    - Return ONLY a valid JSON object with keys that appear in the query.
    - Do NOT assign values to the wrong field.
    - for space name, if the value has 2 words, assign to space_name, if it has 1 word, assign to space_key.
    - If a value could belong to multiple fields, choose the most appropriate.
    - Do not split multi-word values.
    - Return nothing if the query does not specify a field.
    - Generate the timings according to the query e.g. "last week" or "in the last 30 days" should always be converted 
        to an appropriate date string related to today's date calculated in the prompt.
    - also get any important keywords that can help with semantic search (e.g. page titles, topics) mentioned in the 
        query and return them in a "semantic_query" field.
    - for keywords like "unused pages or stale pages", treat them as which are not modified in the last 2 years as a 
        special case and convert that to a lastmodified_timestamp filter with appropriate end date and dont add them in the semantic_query field.
    - We have only limited data from confluence and don't have the body of the pages, so keywords are very important to get the right search results.
    - For "Never modified or never updated", the version number should be 1 and lastmodified_timestamp should be the 
        same as created_timestamp, so convert that to appropriate filters and do not add(e.g., Alice's documents) those keywords in the semantic_query field.
    - If the query mentions a person's name but doesn’t specify "created" or "modified" , assign to created_by only.
    Drop the conjunctions and prepositions and return only the core keywords that can help with semantic search in the "semantic_query" field.
    - return the output in the following JSON format with either value or None and not null for missing fields:
    {{
       "lastmodified_author": "<value or None>",
       "space_name": "<value or None>",
       "lastmodified_timestamp": {{
           "start": "<YYYY-MM-DD or None>",
           "end": "<YYYY-MM-DD or None>"
       }},
       "created_by": "<value or None>",
       "created_timestamp": {{
           "start": "<YYYY-MM-DD or None>",
           "end": "<YYYY-MM-DD or None>"
       }},
       "page_id": "<value or Nonel>",
       "version_number": "<value or None>",
       "space_key": "<value or None>",
       "ancestor_titles": "<value or None>",
       "semantic_query": "<value or None>"
    }}

    Query: "{query}"
    """

    llm_client = AzureOpenAI(
        api_version=api_version_gpt,
        azure_endpoint=endpoint,
        api_key=API_KEY,
    )

    response = llm_client.chat.completions.create(
        model=gpt_model_name,
        response_format={"type": "json_object"},
        messages=[{"role": "user", "content": prompt_1}],
        temperature=0
    )
    if (response.choices[0].message.content.strip() == ""):
        print("GPT-4 returned empty content for filters. Response:", response)
        return {}

    try:
        content = response.choices[0].message.content
        filters = parse_gpt_json(content)
    except Exception as e:
        print("Exception while parsing GPT-4 response. ", e)
        print("Response content:", response.choices[0].message.content)
        filters = {}

    return filters

# Example: perform a query
#user_query = "What are the pages created by Vasanthi before 5 years?"

# -------------------------
# Run semantic + metadata search
# -------------------------
#query_embedding = embed_text(user_query)

#metadata_filters = extract_filters(user_query)
#print("Filters extracted by GPT-4:", metadata_filters)
# Convert to JSON string
#json_output = json.dumps(metadata_filters, indent=2)
#print("JSON output:", json_output)