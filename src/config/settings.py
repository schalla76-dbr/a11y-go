You don't often get email from paritoshdagar@gmail.com. Learn why this is important
CAUTION: This email originated from outside of the organization.
Do not click links or open attachments unless you recognize the sender and know the content is safe.

#
# File: a11y_go/config/settings.py
#
from dataclasses import dataclass

@dataclass
class A11yGoConfig:
# Unity Catalog details
CATALOG: str = "hackathon11_data"
SCHEMA: str = "bright_initiative"

# Table names used by the tools
CUSTOMERS_TABLE: str = f"{CATALOG}.{SCHEMA}.customers"
FEATURES_TABLE: str = f"{CATALOG}.{SCHEMA}.property_accessibility_features"

# Databricks Model & Vector Search Endpoints
DBRX_ENDPOINT: str = "databricks-dbrx-instruct"
VECTOR_SEARCH_ENDPOINT: str = "a11y_go_vector_search_endpoint"
EMBEDDING_ENDPOINT: str = "databricks-bge-large-en"
REVIEWS_INDEX_FULL_NAME: str = f"{CATALOG}.{SCHEMA}.property_reviews_index"

# ==============================================================================
#
# File: a11y_go/tools/profile_tool.py
#
from langchain.tools import BaseTool
from pyspark.sql import SparkSession
import json

class ProfileTool(BaseTool):
name: str = "get_customer_profile"
description: str = "Use this tool to get the profile, including accessibility needs, for a customer using their customer_id."

spark: SparkSession
customers_table: str

def _run(self, customer_id: str) -> str:
"""Retrieves a customer's profile from the customers table."""
try:
profile_df = self.spark.sql(f"SELECT * FROM {self.customers_table} WHERE customer_id = '{customer_id}'")
if profile_df.count() == 0:
return f"Error: No profile found for customer_id '{customer_id}'."
return json.dumps(profile_df.first().asDict(), indent=2)
except Exception as e:
return f"An error occurred while fetching customer profile: {str(e)}"

# ==============================================================================
#
# File: a11y_go/tools/accommodation_tool.py
#
from langchain.tools import BaseTool
from pyspark.sql import SparkSession
from typing import List
import json

class AccommodationTool(BaseTool):
name: str = "find_accessible_accommodations"
description: str = "Use this tool to find and recommend accommodations based on a list of specific accessibility needs."

spark: SparkSession
features_table: str

def _run(self, accessibility_needs: List[str]) -> str:
"""Finds accommodations that have all the specified accessibility features."""
try:
if not isinstance(accessibility_needs, list) or not accessibility_needs:
return "Error: A list of one or more accessibility needs is required."

# Create a filter condition for each need
filter_conditions = " AND ".join([f"array_contains(accessibility_features, '{need}')" for need in accessibility_needs])

query = f"""
SELECT property_id, title FROM {self.features_table}
WHERE {filter_conditions}
LIMIT 5
"""
results_df = self.spark.sql(query)

if results_df.count() == 0:
return f"No accommodations found with all of the following features: {', '.join(accessibility_needs)}"

return json.dumps([row.asDict() for row in results_df.collect()], indent=2)
except Exception as e:
return f"An error occurred while searching for accommodations: {str(e)}"

# ==============================================================================
#
# File: a11y_go/tools/reviews_tool.py
#
from langchain.tools import BaseTool
from databricks.vector_search.client import VectorSearchClient
import json

class CommunityReviewsTool(BaseTool):
name: str = "search_community_reviews"
description: str = "Use this to search for what people are saying about specific accessibility topics. Use it for questions like 'What are the reviews for wheelchair access?' or 'Are there comments about staff helpfulness for deaf guests?'."

vs_endpoint: str
index_name: str

def _run(self, query: str) -> str:
"""Performs a semantic search over the reviews vector index."""
try:
vsc = VectorSearchClient()
results = vsc.get_index(endpoint_name=self.vs_endpoint, index_name=self.index_name).similarity_search(
query=query,
num_results=3,
columns=["property_id", "title", "review_text"]
)
if not results or 'result' not in results or 'data_array' not in results['result']:
return "No relevant reviews found for your query."

return json.dumps(results['result']['data_array'], indent=2)
except Exception as e:
return f"An error occurred during review search: {str(e)}"

# ==============================================================================
#
# File: a11y_go/agents/concierge_agent.py
#
from langchain import hub
from langchain.agents import AgentExecutor, create_react_agent
from langchain_community.chat_models import ChatDatabricks
from pyspark.sql import SparkSession

# Use relative imports, which work perfectly with Databricks Repos
from ..config.settings import A11yGoConfig
from ..tools.profile_tool import ProfileTool
from ..tools.accommodation_tool import AccommodationTool
from ..tools.reviews_tool import CommunityReviewsTool

class ConciergeAgent:
def __init__(self, spark: SparkSession, config: A11yGoConfig = A11yGoConfig()):
self.spark = spark
self.config = config
self.agent_executor = self._initialize_agent()

def _initialize_agent(self):
"""Initializes the LangChain ReAct agent with tools and the DBRX LLM."""
print("Initializing A11y Go Concierge Agent...")

tools = [
ProfileTool(spark=self.spark, customers_table=self.config.CUSTOMERS_TABLE),
AccommodationTool(spark=self.spark, features_table=self.config.FEATURES_TABLE),
CommunityReviewsTool(vs_endpoint=self.config.VECTOR_SEARCH_ENDPOINT, index_name=self.config.REVIEWS_INDEX_FULL_NAME)
]

llm = ChatDatabricks(endpoint=self.config.DBRX_ENDPOINT, max_tokens=500, temperature=0.1)

prompt = hub.pull("hwchase17/react")
agent = create_react_agent(llm, tools, prompt)

return AgentExecutor(agent=agent, tools=tools, verbose=True, handle_parsing_errors=True)

def run(self, user_prompt: str) -> dict:
"""The main entry point to run a query against the agent."""
return self.agent_executor.invoke({"input": user_prompt})

