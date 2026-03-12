import json
from filter_extractor import extract_filters

user_query = "What are the pages created by Vasanthi before 5 years?"

filters = extract_filters(user_query)

print("Filters extracted:")
print(json.dumps(filters, indent=2))