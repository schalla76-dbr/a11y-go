#
# File: notebooks/01_main_agent_workflow.py
#
# This is the primary notebook for interacting with the A11y Go Agent.

# COMMAND ----------
# MAGIC %md
# MAGIC ## A11y Go - Accessible Travel Agent
# MAGIC
# MAGIC This notebook demonstrates how to use the custom agent built in the `a11y_go` library. It leverages Databricks-native tools, including Unity Catalog, Delta Lake, and AI Functions.

# COMMAND ----------
# MAGIC %pip install langchain langchain-community databricks-vectorsearch databricks-sdk

# COMMAND ----------
dbutils.library.restartPython()

# COMMAND ----------
import sys
import os

# --- This is the key to making the library importable in Databricks Repos ---
# Add the repository's root directory to the Python path.
repo_path = os.path.abspath(os.path.join(os.getcwd(), '..'))
if repo_path not in sys.path:
sys.path.append(repo_path)

# Now you can import from your custom library
from a11y_go.agents.concierge_agent import ConciergeAgent
from a11y_go.config.settings import A11yGoConfig

# COMMAND ----------
# Initialize the agent
# The `spark` session is globally available in Databricks notebooks.
config = A11yGoConfig()
agent = ConciergeAgent(spark, config)

# COMMAND ----------
# MAGIC %md
# MAGIC ### Example 1: Multi-Step Query
# MAGIC
# MAGIC Find the accessibility needs for a customer and then find hotels that match those needs.

# COMMAND ----------
prompt_1 = "My customer is Alex (ID: 1). Find some hotels that meet his specific accessibility needs."
result_1 = agent.run(prompt_1)

# COMMAND ----------
# MAGIC %md
# MAGIC ### Example 2: RAG-Based Query
# MAGIC
# MAGIC Use the Vector Search RAG tool to get community insights about a specific accessibility topic.

# COMMAND ----------
prompt_2 = "What do people say in reviews about roll-in showers?"
result_2 = agent.run(prompt_2)

# COMMAND ----------
# Print the final answer for clarity
print("-------------------- FINAL ANSWER --------------------")
print(result_2['output'])
