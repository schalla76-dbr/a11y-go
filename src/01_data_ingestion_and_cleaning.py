# ==============================================================================
# File: databricks_setup/01_data_ingestion_and_cleaning.py
# ==============================================================================
# Description: This notebook ingests the raw CSV data you have uploaded to a
# Databricks Volume, cleans it, and creates three core Delta tables
# in Unity Catalog.
# ==============================================================================

from pyspark.sql.functions import col, lit, regexp_replace
import re

# === Configuration: UPDATE THESE VALUES ===
CATALOG = "hackathon11_data" # Suggestion: Use a descriptive catalog name
SCHEMA = "bright_initiative" # Suggestion: Use a descriptive schema name

# --- Ensure Catalog and Schema Exist ---
spark.sql(f"CREATE CATALOG IF NOT EXISTS `{CATALOG}`")
spark.sql(f"USE CATALOG `{CATALOG}`")
spark.sql(f"CREATE SCHEMA IF NOT EXISTS `{SCHEMA}`")
spark.sql(f"USE SCHEMA `{SCHEMA}`")

# --- Define Paths to Uploaded Data in Volumes ---
# Replace 'your_volume_path' with the actual path in Unity Catalog where you uploaded the files
# For example: /Volumes/hackathon11_data/bright_initiative/raw_data/
VOLUME_PATH = "/Volumes/hackathon11_data/bright_initiative/raw_data"
raw_airbnb_path = f"{VOLUME_PATH}/hackathon11_data.bright_initiative.airbnb_properties_information_csv"
raw_booking_path = f"{VOLUME_PATH}/hackathon11_data.bright_initiative.booking_hotel_listings_csv"
raw_google_path = f"{VOLUME_PATH}/hackathon11_data.bright_initiative.google_maps_businesses_csv"

# --- Ingest & Clean Airbnb Data ---
airbnb_df = spark.read.format("csv").option("header", "true").option("inferSchema", "false").option("multiLine", "true").option("escape", "\"").load(raw_airbnb_path)

# Create a unique, clean property_id from the URL
airbnb_df = airbnb_df.withColumn("property_id", regexp_replace(col("url"), "https://www.airbnb.com/rooms/|\\?.+", "")) \
.withColumn("source", lit("airbnb")) \
.withColumnRenamed("name", "title") \
.withColumnRenamed("rating", "review_score") \
.select("property_id", "title", "description", "address", "review_score", "country", "lat", "lon", "source", "amenities", "reviews", "url")

airbnb_df.write.mode("overwrite").saveAsTable("airbnb_properties")
print("Successfully created and saved 'airbnb_properties' Delta table.")
display(spark.table("airbnb_properties"))


# --- Ingest & Clean Booking.com Data ---
booking_df = spark.read.format("csv").option("header", "true").option("inferSchema", "false").option("multiLine", "true").option("escape", "\"").load(raw_booking_path)
booking_df = booking_df.withColumn("source", lit("booking")) \
.withColumnRenamed("hotel_id", "property_id") \
.withColumn("address", col("location")) \
.withColumnRenamed("most_popular_facilities", "amenities") \
.withColumnRenamed("top_reviews", "reviews") \
.select("property_id", "title", "description", "address", "review_score", "country", "source", "amenities", "reviews", "url") # lat/lon are often missing/malformed

booking_df.write.mode("overwrite").saveAsTable("booking_hotels")
print("Successfully created and saved 'booking_hotels' Delta table.")
display(spark.table("booking_hotels"))

# --- Ingest & Clean Google Places Data ---
google_df = spark.read.format("csv").option("header", "true").option("inferSchema", "false").option("multiLine", "true").option("escape", "\"").load(raw_google_path)
google_df = google_df.withColumn("source", lit("google")) \
.withColumnRenamed("place_id", "property_id") \
.withColumn("title", col("name")) \
.withColumnRenamed("rating", "review_score") \
.withColumnRenamed("hotel_amenities", "amenities") \
.select("property_id", "title", "description", "address", "review_score", "country", "lat", "lon", "source", "amenities", "reviews", "url")

google_df.write.mode("overwrite").saveAsTable("google_places")
print("Successfully created and saved 'google_places' Delta table.")
display(spark.table("google_places"))

# ==============================================================================
# File: databricks_setup/02_feature_engineering.py
# ==============================================================================
# Description: This notebook combines the data sources, then uses a Databricks
# foundation model (DBRX) to extract structured accessibility features.
# ==============================================================================

from pyspark.sql.functions import col, from_json
from pyspark.sql.types import StringType, StructType, StructField, ArrayType

# --- Configuration ---
CATALOG = "hackathon11_data"
SCHEMA = "bright_initiative"
DBRX_ENDPOINT = "databricks-dbrx-instruct" # Use the built-in Foundation Model endpoint
spark.sql(f"USE {CATALOG}.{SCHEMA}")

# --- Combine all properties into a single source-of-truth table ---
unified_df = spark.table("airbnb_properties").select("property_id", "title", "amenities", "reviews") \
.unionByName(spark.table("booking_hotels").select("property_id", "title", "amenities", "reviews")) \
.unionByName(spark.table("google_places").select("property_id", "title", "amenities", "reviews"))

unified_df.write.mode("overwrite").saveAsTable("all_properties_unioned")
print("Created unified properties table.")

# --- Use AI_QUERY for Feature Engineering ---
json_schema = StructType([
StructField("accessibility_features", ArrayType(StringType()), True, {"description": "A list of accessibility features found."})
])

# Use AI_QUERY to call DBRX Instruct and parse the JSON response
query = f"""
CREATE OR REPLACE TABLE property_accessibility_features AS
SELECT
property_id,
title,
AI_QUERY(
'{DBRX_ENDPOINT}',
CONCAT(
'You are an expert data analyst. From the text below, extract accessibility features like "wheelchair accessible", "step-free access", "roll-in shower", "braille signage", "hearing loop", etc. Respond only with a JSON object with one key "accessibility_features" which is a list of strings. If no features are present, return an empty list. Amenities text: ',
amenities
),
'returnType',
'{json_schema.json()}'
) AS extracted_data
FROM all_properties_unioned
WHERE amenities IS NOT NULL
"""
spark.sql(query)
print("Accessibility feature extraction complete.")

# Clean up the final table
final_features_df = spark.table("property_accessibility_features") \
.select("property_id", "title", "extracted_data.accessibility_features") \
.filter(col("accessibility_features").isNotNull())

final_features_df.write.mode("overwrite").saveAsTable("property_accessibility_features")
print("Cleaned accessibility features table saved.")
display(final_features_df)

# ==============================================================================
# File: databricks_setup/03_vector_search_setup.py
# ==============================================================================
# Description: This notebook sets up Databricks Vector Search to enable RAG on
# property reviews.
# ==============================================================================

from databricks.vector_search.client import VectorSearchClient
from pyspark.sql.functions import col, explode, from_json

# --- Configuration ---
CATALOG = "hackathon11_data"
SCHEMA = "bright_initiative"
VECTOR_SEARCH_ENDPOINT = "a11y_go_vector_search_endpoint"
SOURCE_TABLE = f"{CATALOG}.{SCHEMA}.all_properties_unioned"
REVIEWS_FOR_RAG_TABLE = f"{CATALOG}.{SCHEMA}.reviews_for_rag"
INDEX_NAME = f"{CATALOG}.{SCHEMA}.property_reviews_index"
EMBEDDING_ENDPOINT = 'databricks-bge-large-en' # Databricks foundation embedding model

spark.sql(f"USE {CATALOG}.{SCHEMA}")

# --- Prepare Reviews Data ---
# The reviews are in a JSON-like string, so we first parse and then explode.
review_schema = ArrayType(StringType())

reviews_df = spark.table(SOURCE_TABLE) \
.withColumn("reviews_array", from_json(col("reviews"), review_schema)) \
.filter(col("reviews_array").isNotNull()) \
.withColumn("review_text", explode(col("reviews_array"))) \
.select("property_id", "title", "review_text") \
.filter("review_text IS NOT NULL AND length(review_text) > 20") # Filter out empty/short reviews

reviews_df.write.mode("overwrite").saveAsTable(REVIEWS_FOR_RAG_TABLE)
print(f"Created and saved reviews table for RAG: {REVIEWS_FOR_RAG_TABLE}")
spark.sql(f"ALTER TABLE {REVIEWS_FOR_RAG_TABLE} SET TBLPROPERTIES (delta.enableChangeDataFeed = true)")

# --- Create Vector Search Endpoint and Index ---
vsc = VectorSearchClient()

# Check if endpoint exists, otherwise create it
try:
vsc.get_endpoint(name=VECTOR_SEARCH_ENDPOINT)
print(f"Vector Search endpoint '{VECTOR_SEARCH_ENDPOINT}' already exists.")
except Exception:
print(f"Creating Vector Search endpoint '{VECTOR_SEARCH_ENDPOINT}'.")
vsc.create_endpoint(name=VECTOR_SEARCH_ENDPOINT, endpoint_type="STANDARD")

# Create a Delta Sync Index with self-managed embeddings
try:
vsc.create_delta_sync_index(
endpoint_name=VECTOR_SEARCH_ENDPOINT,
source_table_name=REVIEWS_FOR_RAG_TABLE,
index_name=INDEX_NAME,
pipeline_type="TRIGGERED",
primary_key="property_id",
embedding_source_column='review_text',
embedding_model_endpoint_name=EMBEDDING_ENDPOINT
)
print(f"Successfully created Vector Search index '{INDEX_NAME}'.")
except Exception as e:
if "already exists" in str(e):
print(f"Index '{INDEX_NAME}' already exists.")
else:
raise e

