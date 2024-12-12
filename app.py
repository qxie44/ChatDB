import pandas as pd
from flask import Flask, request, jsonify, render_template
import mysql.connector
from pymongo import MongoClient
import spacy
import re
import os
import csv
import json
from werkzeug.utils import secure_filename
from pandas import read_excel
from difflib import get_close_matches
import logging
from datetime import time
from bson import ObjectId
import pymongo
import random

# Load NLP 
nlp = spacy.load("en_core_web_sm")

# Flask setup
app = Flask(__name__)
UPLOAD_FOLDER = 'uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
ALLOWED_EXTENSIONS = {'csv', 'json', 'xlsx'}

# Set up logging for debug
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# MySQL connection setup
mysql_conn = mysql.connector.connect(
    host="localhost",
    user="root",
    password="YourNewSecurePassword",
    database="chatdb_sql"
)

# MongoDB connection setup
mongo_client = MongoClient("mongodb://localhost:27017/")
mongo_db = mongo_client["chatdb_nosql"]

# In-memory column mapping storage
column_mappings = {}

# Helper function to check allowed file types mentioned above
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def home():
    return render_template('index.html')

uploaded_datasets = {}

# File upload route
@app.route('/upload', methods=['POST'])
def upload_file():
    try:
        file = request.files.get('file')
        if not file:
            return jsonify({"error": "No file part"}), 400
        if file.filename == '':
            return jsonify({"error": "No selected file"}), 400
        if allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(file_path)

            # Placeholder for dataset preview
            preview = None
            message = ""

            # Process file based on type
            if filename.endswith('.csv'):
                message = insert_csv_to_mysql(file_path)
                df = pd.read_csv(file_path)
            elif filename.endswith('.json'):
                message = insert_json_to_mongodb(file_path)
                with open(file_path, 'r') as json_file:
                    data = json.load(json_file)
                    df = pd.DataFrame(data)
            elif filename.endswith('.xlsx'):
                message = insert_xlsx_to_mysql(file_path)
                df = pd.read_excel(file_path)
            else:
                return jsonify({"error": "Unsupported file type"}), 400
            
            dataset_name = os.path.splitext(filename)[0]
            uploaded_datasets[dataset_name] = {
                "data": df,
                "columns": df.columns.tolist(),  # Store column names
            }
            
            if 'transaction_date' in df.columns and pd.api.types.is_datetime64_any_dtype(df['transaction_date']):
                df['transaction_date'] = df['transaction_date'].dt.date  # Truncate to date
                logger.debug(f"Truncated 'transaction_date' to date format.")

            # Generate a preview (convert datetime/time columns to strings)
            preview = df.head()
            for col in preview.columns:
                # Handle datetime columns
                if pd.api.types.is_datetime64_any_dtype(preview[col]):
                    preview[col] = preview[col].dt.strftime('%Y-%m-%d %H:%M:%S')
                # Handle timedelta columns
                elif pd.api.types.is_timedelta64_dtype(preview[col]):
                    preview[col] = preview[col].astype(str)
                # Handle time objects 
                elif preview[col].apply(lambda x: isinstance(x, time)).any():
                    preview[col] = preview[col].astype(str)
                # Parse ambiguous columns
                elif preview[col].dtype == 'object':
                    try:
                        preview[col] = pd.to_datetime(preview[col]).dt.strftime('%Y-%m-%d %H:%M:%S')
                    except Exception:
                        pass  # Skip conversion if parsing fails

            # Ensure all data is JSON 
            preview = preview.astype(str)

            return jsonify({
                "success": message,
                "preview": preview.to_dict(orient='records'),
                "columns": df.columns.tolist(),  # Send column names to frontend
            }), 200
        else:
            return jsonify({"error": "Invalid file type"}), 400
    except Exception as e:
        # Log error
        app.logger.error(f"Error while processing upload: {e}")
        return jsonify({"error": "Internal server error: " + str(e)}), 500


@app.route('/get_columns', methods=['POST'])
def get_columns():
    data = request.json
    dataset_name = data.get("dataset")

    if dataset_name not in uploaded_datasets:
        return jsonify({"error": f"Dataset '{dataset_name}' not found."}), 400

    columns = uploaded_datasets[dataset_name]["columns"]
    return jsonify(columns), 200




# Insert CSV data into MySQL
def insert_csv_to_mysql(file_path):
    with open(file_path, 'r') as file:
        csv_reader = csv.reader(file)
        headers = next(csv_reader)
        normalized_headers = [f"`{header.strip().replace(' ', '_').replace('-', '_').lower()}`"  for header in headers]
        logger.debug(f"Normalized headers: {normalized_headers}")
        table_name = os.path.splitext(os.path.basename(file_path))[0]
        cursor = mysql_conn.cursor()

        # Create a dynamic table 
        create_table_query = f"CREATE TABLE IF NOT EXISTS {table_name} ({', '.join([f'{col} VARCHAR(255)' for col in headers])})"
        cursor.execute(create_table_query)

        for row in csv_reader:
            insert_query = f"INSERT INTO {table_name} ({', '.join(headers)}) VALUES ({', '.join(['%s'] * len(row))})"
            cursor.execute(insert_query, row)
        mysql_conn.commit()

        # Generate column mapping
        column_mappings[table_name] = {header.lower(): header for header in headers}
        logger.debug(f"Column mappings for {table_name}: {column_mappings[table_name]}")
        return f"Table '{table_name}' created and data inserted successfully!"

# Insert JSON data into MongoDB
def insert_json_to_mongodb(file_path):
    try:
        with open(file_path, 'r') as file:
            data = json.load(file)
            logger.debug(f"Loaded JSON data from {file_path}")

            # Extract collection name from the file name
            collection_name = os.path.splitext(os.path.basename(file_path))[0]
            logger.debug(f"Target MongoDB collection: {collection_name}")

            collection = mongo_db[collection_name]

            # Clear the collection to avoid duplicates
            collection.delete_many({})
            logger.debug(f"Cleared collection '{collection_name}' before insertion.")

            # Avoid duplicates by upserting data
            if isinstance(data, list):
                operations = []
                for record in data:
                    if "_id" not in record:
                        record["_id"] = ObjectId()
                    operations.append(
                        pymongo.UpdateOne(
                            {"_id": record["_id"]},
                            {"$set": record},
                            upsert=True
                        )
                    )
                if operations:
                    collection.bulk_write(operations)
                    logger.debug(f"Inserted/Updated {len(operations)} documents into collection '{collection_name}'")
            else:
                if "_id" not in data:
                    data["_id"] = ObjectId()
                collection.update_one(
                    {"_id": data["_id"]},
                    {"$set": data},
                    upsert=True
                )

            # Generate normalized headers and column mapping for query
            sample_document = data[0] if isinstance(data, list) else data
            headers = sample_document.keys()
            normalized_headers = [header.strip().replace(" ", "_").lower() for header in headers]
            column_mappings[collection_name] = {
                normalized_header: original_header 
                for normalized_header, original_header in zip(normalized_headers, headers)
            }

            # Log the normalized headers and generated column mapping
            logger.debug(f"Normalized headers for collection '{collection_name}': {normalized_headers}")
            logger.debug(f"Generated column mapping for collection '{collection_name}': {column_mappings[collection_name]}")

            return f"Collection '{collection_name}' created and data inserted successfully!"
    except Exception as e:
        logger.error(f"Error while inserting JSON to MongoDB: {str(e)}")
        raise

# Insert XLSX data into MySQL
def insert_xlsx_to_mysql(file_path):
    try:
        # Read the Excel file
        df = pd.read_excel(file_path)
        logger.debug(f"Loaded Excel file from {file_path}")

        # Extract table name from file name
        table_name = os.path.splitext(os.path.basename(file_path))[0]
        logger.debug(f"Target MySQL table: {table_name}")

        cursor = mysql_conn.cursor()

        # Normalize headers
        headers = df.columns
        normalized_headers = [header.strip().replace(" ", "_").lower() for header in headers]
        logger.debug(f"Normalized headers for table '{table_name}': {normalized_headers}")

        # Create dynamiic table with normalized headers
        create_table_query = f"CREATE TABLE IF NOT EXISTS {table_name} ({', '.join([f'{col} VARCHAR(255)' for col in normalized_headers])})"
        logger.debug(f"Create table query: {create_table_query}")
        cursor.execute(create_table_query)

        # Insert data into the table
        for _, row in df.iterrows():
            insert_query = f"INSERT INTO {table_name} ({', '.join(normalized_headers)}) VALUES ({', '.join(['%s'] * len(row))})"
            cursor.execute(insert_query, tuple(row))
            logger.debug(f"Executed insert query for row: {row.to_dict()}")

        # Commit the transaction
        mysql_conn.commit()
        logger.debug(f"Data successfully inserted into table '{table_name}'")

        # Generate column mapping
        column_mappings[table_name] = {
            normalized_header: original_header
            for normalized_header, original_header in zip(normalized_headers, headers)
        }
        logger.debug(f"Generated column mapping for table '{table_name}': {column_mappings[table_name]}")

        return f"Table '{table_name}' created and data inserted successfully!"
    except Exception as e:
        logger.error(f"Error while inserting XLSX to MySQL: {str(e)}")
        return str(e)

# Run query route for SQL or MongoDB
@app.route('/run_query', methods=['POST'])
def run_query():
    data = request.json
    logger.debug(f"Received data: {data}")

    # User_query is assigned a default value
    user_query = data.get('query', '').strip()
    logger.debug(f"User query: {user_query}")
    collection_name = data.get('table', '').replace(' ', '_')
    database_type = data.get('database_type', 'SQL')

    # Handle "example queries" input
    if user_query.lower().startswith("example queries"):
        # Generate three example options and allow users to input desired option
        options = {
            1: "Group By Query",
            2: "Aggregation Query",
            3: "Sort Query"
        }
        return jsonify({
            "example_queries": options
        }), 200

    # selection of example query
    if user_query.lower().startswith("example option"):
        try:
            # Extract the option number from the query
            option = int(user_query.split()[-1])
        except ValueError:
            return jsonify({"error": "Invalid option number provided"}), 400

        if option not in [1, 2, 3]:
            return jsonify({"error": "Invalid example query option"}), 400

        table_name = data.get('table', '').replace(' ', '_')
        if not table_name:
            return jsonify({"error": "No dataset selected for generating examples."}), 400

        column_mapping = column_mappings.get(table_name, {})
        if not column_mapping:
            return jsonify({"error": "No columns available for the selected dataset."}), 400

        dataset_preview = uploaded_datasets.get(collection_name, {}).get("data")
        if dataset_preview is None:
            return jsonify({"error": "Dataset preview unavailable"}), 400
        logger.debug("preview part done")

        # Generate the query based on the selected option
        try:
            if database_type == 'NoSQL':
                logger.debug("identified as nosql")
                if option == 1:
                    logger.debug("nosql option1")
                    query = generate_mongo_group_by_query(collection_name, dataset_preview)
                elif option == 2:
                    logger.debug("nosql option2")
                    query = generate_mongo_aggregation_query(collection_name, dataset_preview)
                elif option == 3:
                    logger.debug("nosql option3")
                    query = generate_mongo_sort_query(collection_name, dataset_preview)

                if not query:
                    return jsonify({"error": "No valid columns available for the selected query."}), 400

                # Execute the generated MongoDB query
                headers, results = execute_mongo_query(collection_name, query)

                return jsonify({
                    "generated_example": json.dumps(query, indent=2),
                    "headers": headers,
                    "results": results
                }), 200
            if option == 1:
                query = generate_group_by_query(column_mapping, table_name, dataset_preview)
            elif option == 2:
                query = generate_aggregation_query(column_mapping, table_name, dataset_preview)
            elif option == 3:
                query = generate_sort_query(column_mapping, table_name, dataset_preview)

            if not query:
                return jsonify({"error": "No valid columns found to generate the selected query."}), 400

            # Proceed to execute the generated query
            headers, results = execute_query(query, None, 'SQL', table_name)

            # Return both the query and the results
            return jsonify({
                "generated_example": query,
                "headers": headers,
                "results": results
            }), 200
        
        except ValueError as ve:
            # Return the error message to the UI
            logger.error(f"No numeric columns: {str(ve)}")
            return jsonify({"error": f"Failed to generate example query, no numeric columns: {str(ve)}"}), 400

        except Exception as e:
            logger.error(f"Failed to generate or execute example query: {str(e)}")
            return jsonify({"error": f"Failed to generate or execute example query: {str(e)}"}), 500

    user_query = data.get('query', '').strip()
    logger.debug(f"User query: {user_query}")

    # "reset all" command to drop tables/collections
    if user_query == "reset all":
        try:
            mysql_message = drop_all_mysql_tables()
            mongodb_message = drop_all_mongodb_collections()
            return jsonify({"success": f"{mysql_message} {mongodb_message}"}), 200
        except Exception as e:
            return jsonify({"error": f"Reset operation failed: {str(e)}"}), 500

    # Extract table name and database type
    table_name = data.get('table', '').replace(' ', '_')  # Normalize table name
    logger.debug(f"Table name: {table_name}")

    database_type = data.get('database_type', 'SQL')  # Default to SQL if not provided
    logger.debug(f"Database type: {database_type}")

    # Validate table/collection name
    if not table_name:
        logger.error("Table/Collection name is required but missing.")
        return jsonify({"error": "Table/Collection name is required"}), 400

    # Check if the table/collection exists in column_mappings
    if table_name not in column_mappings:
        available_datasets = ", ".join(column_mappings.keys())
        logger.error(f"Table/Collection '{table_name}' does not exist. Available datasets: {available_datasets}")
        return jsonify({
            "error": f"Table/Collection '{table_name}' does not exist. Available datasets: {available_datasets}"
        }), 400

    try:
        # Generate the query
        query = generate_query(user_query, None, column_mappings.get(table_name), database_type, table_name)
        logger.debug(f"Generated query: {query}")

        # Execute the query based on database type
        headers, results = execute_query(query, None, database_type, table_name)

        # Include MongoDB query as a string 
        if database_type == 'NoSQL':
            query_string = json.dumps(query, indent=2)  # Convert pipeline to a JSON string
        else:
            query_string = query  # For SQL, use the raw query string

        # Include the query in the output
        return jsonify({
            "query": query_string, 
            "headers": headers,
            "results": results
        }), 200
    except Exception as e:
        logger.error(f"Query execution failed: {str(e)}")
        return jsonify({"error": f"Query execution failed: {str(e)}"}), 400


# Generate the random query depending on which number the user selects
# Option 1
def generate_group_by_query(column_mapping, table_name, dataset_preview):
    # Convert column names to lowercase
    dataset_preview.columns = [col.lower() for col in dataset_preview.columns]

    # Prefer categorical columns
    categorical_columns = dataset_preview.select_dtypes(include=['object', 'category']).columns.tolist()
    
    # If no categorical columns, use first column
    group_column = categorical_columns[0] if categorical_columns else dataset_preview.columns[0]

    query = f"SELECT {group_column}, COUNT(*) AS count FROM {table_name} GROUP BY {group_column}"

    return query

# Option 2
def generate_aggregation_query(column_mapping, table_name, dataset_preview):
    # Convert column names to lowercase for consistency
    dataset_preview.columns = [col.lower() for col in dataset_preview.columns]

    # Exclude the 'id' column from selection (assuming column is named 'id' or similar)
    non_id_columns = [col for col in dataset_preview.columns if 'id' not in col.lower()]

    # Select columns to use for query
    categorical_columns = dataset_preview.select_dtypes(include=['object', 'category']).columns.tolist()
    categorical_columns = [col for col in categorical_columns if col in non_id_columns]
    group_column = categorical_columns[0] if categorical_columns else (non_id_columns[0] if non_id_columns else None)
    numeric_columns = dataset_preview.select_dtypes(include=['int64', 'float64']).columns.tolist()
    numeric_columns = [col for col in numeric_columns if col in non_id_columns]

    if not group_column or not numeric_columns:
        logger.error("No valid columns available for aggregation query.")
        return None

    # Construct the aggregation query using non-ID columns
    agg_column = numeric_columns[0]
    query = f"SELECT {group_column}, SUM({agg_column}) AS total_{agg_column} FROM {table_name} GROUP BY {group_column}"
    
    return query

# Option 3
def generate_sort_query(column_mapping, table_name, dataset_preview):
    # Convert column names to lowercase
    dataset_preview.columns = [col.lower() for col in dataset_preview.columns]

    # Find numeric columns for sorting
    numeric_columns = dataset_preview.select_dtypes(include=['int64', 'float64']).columns.tolist()

    # If no numeric columns, default to first column
    if not numeric_columns:
        sort_column = dataset_preview.columns[0]
    else:
        # Prefer a column that isn't the first column (usually an ID)
        sort_column = numeric_columns[0] if len(numeric_columns) == 1 else numeric_columns[1]

    # Construct sorting query
    query = f"SELECT * FROM {table_name} ORDER BY {sort_column} ASC LIMIT 100"
    return query


# Generate SQL or MongoDB query based on user input
def generate_query(user_input, df, column_mapping, database_type='SQL', table_name=None):
    logger.debug(f"Generating query. User input: {user_input}")
    logger.debug(f"Column mapping: {column_mapping}")
    if not column_mapping:
        raise ValueError(f"Column mapping for the table is missing or empty.")
    if not table_name:
        raise ValueError("Table name is required but not provided.")
    if database_type == 'SQL':
        return generate_sql_query(user_input, df, column_mapping, table_name)
    elif database_type == 'NoSQL':
        return generate_mongo_query(user_input, df, column_mapping, table_name)

# Execute SQL or MongoDB query
def execute_query(query, df, database_type='SQL', table_name=None):
    logger.debug(f"Executing query: {query}")
    if database_type == 'SQL':
        cursor = mysql_conn.cursor(buffered=True)
        try:
            cursor.execute(query)
            rows = cursor.fetchall()

            # Extract column names with what is selected
            headers = [desc[0] for desc in cursor.description] if cursor.description else []

            # Debug logs to verify the headers and rows
            logger.debug(f"Fetched headers: {headers}")
            logger.debug(f"Fetched rows: {rows[:5]}")  # Display only the first 5 rows

            return headers, rows
        except mysql.connector.Error as err:
            raise ValueError(f"MySQL Error: {err}")
        finally:
            cursor.close()
    elif database_type == 'NoSQL':
        if not table_name:
            raise ValueError("MongoDB collection name must be provided.")
        return execute_mongo_query(table_name, query)

# Dropping tables

def drop_all_mysql_tables():
    cursor = mysql_conn.cursor()
    # Query to get all table names
    cursor.execute("SHOW TABLES")
    tables = cursor.fetchall()
    
    # Disable foreign key checks and drop all tables
    cursor.execute("SET FOREIGN_KEY_CHECKS = 0;")
    for table in tables:
        cursor.execute(f"DROP TABLE IF EXISTS {table[0]};")
    cursor.execute("SET FOREIGN_KEY_CHECKS = 1;")
    mysql_conn.commit()
    return "All MySQL tables dropped successfully!"

def drop_all_mongodb_collections():
    collections = mongo_db.list_collection_names()
    for collection in collections:
        mongo_db.drop_collection(collection)
    return "All MongoDB collections dropped successfully!"

# SQL Functions

# Detects any aggegration key words in user input and return the corresponding aggregation
def detect_aggregation(user_input):
    logger.debug(f"Detected aggregation function: {user_input}")
    if "total" in user_input.lower() or "sum" in user_input.lower():
        return "SUM"
    elif "average" in user_input.lower() or "avg" in user_input.lower():
        return "AVG"
    elif "maximum" in user_input.lower():
        return "MAX"
    elif "minimum" in user_input.lower():
        return "MIN"
    elif "count" in user_input.lower():
        return "COUNT"
    return None

# Detect columns
def detect_columns(user_input, doc, column_mapping):
    logger.debug(f"Detecting columns in user query: {user_input}")
    logger.debug(f"Available columns: {list(column_mapping.keys())}") 
    detected_columns = [] 

    for token in doc:
        user_friendly_col = token.text.lower().replace(" ", "_")
        if user_friendly_col in column_mapping:
            detected_columns.append(column_mapping[user_friendly_col])
        else:
            # Use string similarity to find the closest match
            matches = get_close_matches(user_friendly_col, column_mapping.keys(), n=1, cutoff=0.8)
            if matches:
                detected_columns.append(column_mapping[matches[0]])
    
    return detected_columns if detected_columns else None

# Looks for group by pattern  in user input
def detect_grouping(user_input, column_mapping):
    logger.debug(f"Processing grouping in user query: {user_input}")

    # Handle "group by" explicitly
    if "group by" in user_input.lower():
        group_by_part = user_input.lower().split("group by", 1)[1].strip()
        group_by_doc = nlp(group_by_part)
        detected_group = detect_columns(group_by_part, group_by_doc, column_mapping)
        logger.debug(f"Detected grouping columns (explicit 'group by'): {detected_group}")
        return detected_group

    # Look for phrases like "by <column>" to infer grouping
    match = re.search(r"by\s+(\w+)", user_input, re.IGNORECASE)
    if match:
        grouping_col = match.group(1).lower().replace(" ", "_")
        logger.debug(f"Detected potential grouping phrase: 'by {grouping_col}'")
        if grouping_col in column_mapping:
            logger.debug(f"Grouping column '{grouping_col}' found in column mapping.")
            return column_mapping[grouping_col]
        else:
            logger.debug(f"Grouping column '{grouping_col}' not found in column mapping.")
    else:
        logger.debug("No 'by <column>' pattern found in user query.")

    # If no grouping detected
    logger.debug("No grouping columns detected.")
    return None

# Detects the limit in user input
def detect_limit(user_input):
    limit_pattern = re.search(r"limit (\d+)", user_input, re.IGNORECASE)
    if limit_pattern:
        return int(limit_pattern.group(1))
    return None

# Detect where condition and maps natural language to operator
def detect_where_condition(user_input):
    logger.debug("Inside where function now")

    # Match conditions with a variety of operators
    matches = re.findall(
        r"(\w+)\s*(is|=|equal to|not equal to|!=|like|>|<|>=|<=|greater than|less than|at least|at most)\s*['\"]?([^'\"]+)['\"]?",
        user_input,
        re.IGNORECASE
    )
    logger.debug(f"Matched conditions: {matches}")

    # Detect logical operators
    logical_ops = re.findall(r"(and|or)", user_input, re.IGNORECASE)
    logger.debug(f"Detected logical operators: {logical_ops}")
    operator_mapping = {
        "is": "=",
        "equal to": "=",
        "not equal to": "!=",
        "!=": "!=",
        "like": "LIKE",
        ">": ">",
        "<": "<",
        ">=": ">=",
        "<=": "<=",
        "greater than": ">",
        "less than": "<",
        "at least": ">=",
        "at most": "<="
    }
    conditions = []
    for column, operator, value in matches:
        logger.debug(f"Processing condition: {column} {operator} {value}")

        # Map the operator
        sql_operator = operator_mapping.get(operator.lower(), operator)

        # Cast value to number if possible
        try:
            value = float(value) if value.replace('.', '', 1).isdigit() else value
        except ValueError:
            pass  # Keep as string if conversion fails

        # Append condition with proper quoting for strings
        if isinstance(value, (int, float)):
            conditions.append(f"{column} {sql_operator} {value}")
        else:
            conditions.append(f"{column} {sql_operator} '{value}'")
    logger.debug(f"Constructed conditions: {conditions}")

    if not conditions:
        logger.debug("No conditions detected in the query.")
        return None

    # Combine conditions with logical operators
    where_clause = conditions[0]
    for i, logical_op in enumerate(logical_ops):
        if i + 1 < len(conditions):  # Ensure there's a matching condition
            where_clause += f" {logical_op.upper()} {conditions[i + 1]}"
    logger.debug(f"Constructed WHERE clause: {where_clause}")

    return where_clause.strip() if where_clause else None



# Detect sorting in user input and sorts in asc or desc order depending on natural language
def detect_sorting(user_input):
    sort_order = None  # Initialize sort_order to None

    if "desc" in user_input.lower() or "descending" in user_input.lower():
        sort_order = "DESC"
    elif "asc" in user_input.lower() or "ascending" in user_input.lower():
        sort_order = "ASC"
    
    logger.debug(f"Detected sorting order: {sort_order}")  # Log after determining sort order
    return sort_order

# Detect having condition after group by 
def detect_having_condition(user_input, detected_group, detected_aggregation, aggregation_alias):
    user_input_lower = user_input.lower().strip()
    logger.debug(f"Checking HAVING condition in query: {user_input}")
    logger.debug(f"'having' in input: {'having' in user_input_lower}")

    if "having" in user_input_lower:
        # Handle explicit conditions
        match = re.search(
            r"having\s+(sum|count|avg|max|min)?\s*\(?\s*([a-zA-Z_]+)?\)?\s*(>=|<=|>|<|=|!=|not equal to|more than|greater than|less than|fewer than)?\s*(\d+)",
            user_input_lower
        )
        if match:
            aggregation = match.group(1) or (detected_aggregation.upper() if detected_aggregation else "COUNT")
            column = match.group(2) or (detected_group[0] if detected_group else None)
            operator = match.group(3)
            value = match.group(4)

            # Check if `column` or `detected_group` is valid
            if not column:
                logger.error("No valid column found for HAVING condition. Ensure grouping is properly detected.")
                return None

            # Map user-friendly operators to SQL
            operator_mapping = {
                "more than": ">",
                "greater than": ">",
                "less than": "<",
                "fewer than": "<",
                "not equal to": "!="
            }
            sql_operator = operator_mapping.get(operator.lower(), operator)

            # Alias for aggregation
            alias = aggregation_alias.get(aggregation.upper(), aggregation.lower() + "_qty")
            logger.debug(f"Constructed HAVING condition: {alias} {sql_operator} {value}")
            return f"{alias} {sql_operator} {value}"

        # Handle implicit conditions like "having more than 1000"
        match = re.search(r"having\s+(more|greater|less|fewer)?\s*(than|equals)?\s*(\d+)", user_input_lower)
        if match:
            operator_mapping = {
                "morethan": ">",
                "greaterthan": ">",
                "lessthan": "<",
                "fewerthan": "<",
                "equals": "="
            }

            # Extract operator and value
            operator_key = (match.group(1) or "") + (match.group(2) or "")
            sql_operator = operator_mapping.get(operator_key.replace(" ", "").lower(), ">")
            value = match.group(3)

            # Use detected aggregation alias or fallback to COUNT
            aggregation = detected_aggregation.upper() if detected_aggregation else "COUNT"
            alias = aggregation_alias.get(aggregation, aggregation.lower() + "_qty")
            logger.debug(f"Constructed implicit HAVING condition: {alias} {sql_operator} {value}")
            return f"{alias} {sql_operator} {value}"

    logger.debug("No HAVING condition detected.")
    return None

# Detect top # in user input
def detect_top_n(user_input):
    match = re.search(r"top (\d+)", user_input, re.IGNORECASE)
    if match:
        return int(match.group(1))
    return None

# Generate the SQL query output by putting together all of the functions in sql order
def generate_sql_query(user_input, df, column_mapping, table_name):
    doc = nlp(user_input)
    aggregation_alias = {
        "SUM": "sum_qty",
        "AVG": "avg_qty",
        "MAX": "max_qty",
        "MIN": "min_qty",
        "COUNT": "count_qty"
    }

    # Detect entities
    detected_aggregation = detect_aggregation(user_input)
    detected_column = detect_columns(user_input, doc, column_mapping)
    detected_group = detect_grouping(user_input, column_mapping)
    logger.debug(f"Detected grouping column: {detected_group}")
    logger.debug(f"Detected aggregation column: {detected_column}")
    sort_order = detect_sorting(user_input)
    logger.debug("sort done, going into where")
    where_conditions = detect_where_condition(user_input)
    logger.debug("where done going into top n")
    top_n = detect_top_n(user_input)
    logger.debug("top n done going into limit")
    limit = detect_limit(user_input)

    # Ensure detected_column and detected_group are unique and non-redundant
    if detected_column:
        detected_column = list(set(detected_column))
    if detected_group:
        detected_group = list(set(detected_group))

    # Handle HAVING conditions
    having_condition = detect_having_condition(
        user_input, 
        detected_group, 
        detected_aggregation, 
        aggregation_alias
    )
    logger.debug(f"Detected HAVING condition: {having_condition}")

    # Construct SELECT clause
    select_columns = "*"
    if detected_group:
        group_by_column = ', '.join(detected_group)  # Ensure grouping columns are comma-separated
        if detected_aggregation:
            alias = aggregation_alias.get(detected_aggregation, "agg_qty")
            aggregation_column = next(
                (col for col in detected_column if col not in detected_group),
                detected_column[0] if detected_column else None
            )
            if aggregation_column:  # Ensure aggregation_column exists
                select_columns = f"{group_by_column}, {detected_aggregation}({aggregation_column}) AS {alias}"
            else:
                select_columns = group_by_column
        else:
            select_columns = group_by_column
    elif detected_column:
        select_columns = ', '.join(detected_column)

    # Build the SQL query
    query = f"SELECT {select_columns} FROM {table_name}"
    if where_conditions:
        query += f" WHERE {where_conditions}"
    if detected_group:
        query += f" GROUP BY {', '.join(detected_group)}"
    if having_condition:
        query += f" HAVING {having_condition}"
    if sort_order:
        sort_column = (
            aggregation_alias.get(detected_aggregation, 
                                  detected_column[0] if detected_column else "1")
        )
        query += f" ORDER BY {sort_column} {sort_order}"
    if top_n or limit:
        query += f" LIMIT {top_n or limit}"
    else:
        query += " LIMIT 100"  # Default to 100 if no explicit limit

    # Debug log for the constructed query
    logger.debug(f"SQL Query being constructed: {query}")

    # Final validation
    if not detected_column and not detected_group:
        raise ValueError("No valid columns or groupings detected in the query.")

    return query


# Mongodb function portino

# Detect the aggregation in natural language
def mongo_detect_aggregation(user_input):
    if "total" in user_input.lower() or "sum" in user_input.lower():
        return "SUM"
    elif "average" in user_input.lower() or "avg" in user_input.lower():
        return "AVG"
    elif "maximum" in user_input.lower():
        return "MAX"
    elif "minimum" in user_input.lower():
        return "MIN"
    elif "count" in user_input.lower():
        return "COUNT"
    return None

# Detect the columns matching to file in input
def mongo_detect_columns(user_input, doc, column_mapping):
    detected_columns = []
    for token in doc:
        user_friendly_col = token.text.lower().replace(" ", "_")
        if user_friendly_col in column_mapping.values():
            detected_columns.append(user_friendly_col)
    return detected_columns if detected_columns else None

# Detect if there is a group by clause in input
def mongo_detect_grouping(user_input, column_mapping):
    if "group by" in user_input.lower():
        group_by_part = user_input.lower().split("group by", 1)[1].strip()
        group_by_doc = nlp(group_by_part)
        return mongo_detect_columns(group_by_part, group_by_doc, column_mapping)
    return None

# Detects the where statement and matches natural langauge to operators
def mongo_detect_where_condition(user_input):
    # Match conditions with a variety of operators
    matches = re.findall(
        r"(\w+)\s*(is|=|equal to|not equal to|!=|like|>|<|>=|<=|greater than|less than|at least|at most)\s*['\"]?([^'\"]+)['\"]?",
        user_input,
        re.IGNORECASE
    )
    # Map user-friendly operators to MongoDB operators
    operator_mapping = {
        "is": "$eq",
        "=": "$eq",
        "equal to": "$eq",
        "not equal to": "$ne",
        "!=": "$ne",
        "like": "$regex",
        ">": "$gt",
        "<": "$lt",
        ">=": "$gte",
        "<=": "$lte",
        "greater than": "$gt",
        "less than": "$lt",
        "at least": "$gte",
        "at most": "$lte"
    }
    conditions = {}
    for column, operator, value in matches:
        # Map the detected operator to MongoDB syntax
        mongo_operator = operator_mapping.get(operator.lower(), "$eq")

        # Cast value to a number if possible
        try:
            if value.isdigit():
                value = int(value)
            else:
                value = float(value)
        except ValueError:
            pass  # Keep the value as a string if it cannot be converted

        if mongo_operator == "$regex":
            conditions[column] = {"$regex": value, "$options": "i"}
        else:
            conditions[column] = {mongo_operator: value}

    return conditions if conditions else None

# Detect sorting 
def mongo_detect_sort_column(user_input, column_mapping):
    match = re.search(r"sort by (\w+)", user_input, re.IGNORECASE)
    if match:
        sort_column = match.group(1).lower()  # Extract the column name
        # Check if the detected column exists in the column mapping
        if sort_column in column_mapping:
            return column_mapping[sort_column]
    return None

# Generate the mongo query by breaking up the user input and returning the functions
def generate_mongo_query(user_input, df, column_mapping, table_name=None):
    doc = nlp(user_input)

    # Detect query components
    detected_columns = detect_columns(user_input, doc, column_mapping)
    aggregation = detect_aggregation(user_input)
    where_conditions = mongo_detect_where_condition(user_input)  # Detect filter conditions
    sort_column = mongo_detect_sort_column(user_input, column_mapping)  # Detect sort column
    sort_order = detect_sorting(user_input)  # Detect sort order (ASC/DESC)
    limit = detect_limit(user_input)

    # Build MongoDB pipeline
    pipeline = []
    if where_conditions:
        pipeline.append({"$match": where_conditions})

    # Aggregation Stage
    if aggregation and detected_columns:
        group_field = detected_columns[1] if len(detected_columns) > 1 else None  # Use second column for grouping if available
        # Map aggregation type to MongoDB operators
        aggregation_operator = {
            "sum": "$sum",
            "avg": "$avg",
            "max": "$max",
            "min": "$min",
            "count": "$sum"  # Use `$sum: 1` for counting
        }
        
        # Get the corresponding MongoDB operator
        operator = aggregation_operator.get(aggregation.lower(), "$sum")  # Default to `$sum` if aggregation not mapped
        
        group_stage = {
            "_id": f"${group_field}" if group_field else None,  # Group by detected field if applicable
            f"{aggregation.lower()}_{detected_columns[0]}": {
                operator: 1 if aggregation.lower() == "count" else f"${detected_columns[0]}"
            }
        }
        pipeline.append({"$group": group_stage})

        # Include aggregated fields dynamically
        projection_stage = {
            f"{aggregation.lower()}_{detected_columns[0]}": 1,  
            "_id": 1 if group_field else 0  # Include `_id` only if grouping is applied
        }

        # Fields in where or sort
        if where_conditions:
            projection_stage.update({field: 1 for field in where_conditions.keys()})
        if sort_column:
            projection_stage[sort_column] = 1
        pipeline.append({"$project": projection_stage})

    # Sorting Stage
    if sort_column and sort_order:
        pipeline.append({"$sort": {sort_column: 1 if sort_order.lower() == "asc" else -1}})

    # Limit Stage
    if limit:
        pipeline.append({"$limit": limit})

    # Projection Stage for specific columns (non-aggregated queries)
    if detected_columns and not aggregation:
        projection_stage = {col: 1 for col in detected_columns}
        projection_stage["_id"] = 0  # Exclude `_id` by default

        # Dynamically include fields used in where or sort
        if where_conditions:
            projection_stage.update({field: 1 for field in where_conditions.keys()})
        if sort_column:
            projection_stage[sort_column] = 1

        pipeline.append({"$project": projection_stage})
    return pipeline


# Execute the query and return results
def execute_mongo_query(table_name, pipeline):
    try:
        if table_name not in mongo_db.list_collection_names():
            raise ValueError(f"Collection '{table_name}' does not exist in the database.")

        collection = mongo_db[table_name]
        results = list(collection.aggregate(pipeline))

        # Convert ObjectId to string for JSON serialization
        for result in results:
            for key, value in result.items():
                if isinstance(value, ObjectId):
                    result[key] = str(value)

        headers = list(results[0].keys()) if results else []
        return headers, results
    except Exception as e:
        logger.error(f"MongoDB query execution failed: {e}")
        raise ValueError(f"MongoDB Error: {e}")
    
# Exclude id columns
def filter_columns_exclude_id(columns):
    return [col for col in columns if "id" not in col.lower()]
    

# Example of mongo query from option 1
def generate_mongo_group_by_query(collection_name, dataset_preview):
    # Detect a categorical column for grouping
    categorical_columns = filter_columns_exclude_id(
        dataset_preview.select_dtypes(include=['object', 'category']).columns.tolist()
    )
    #categorical_columns = dataset_preview.select_dtypes(include=['object', 'category']).columns.tolist()

    #group_field = categorical_columns[0] if categorical_columns else dataset_preview.columns[0]
    # Log dataset columns and categorical columns for debugging
    logger.debug(f"Dataset preview columns: {list(dataset_preview.columns)}")
    logger.debug(f"Categorical columns available for grouping: {categorical_columns}")

    # Choose a random column from categorical_columns or dataset_preview.columns
    group_field = None
    if categorical_columns:
        random.seed()
        group_field = random.choice(categorical_columns)  # Randomly pick a categorical column
    else:
        random.seed()
        group_field = random.choice(dataset_preview.columns)  # Randomly pick any column if no categorical columns exist
    
    # Log the chosen grouping column
    logger.debug(f"Chosen grouping column: {group_field}")

    # Check if the column has unique values
    unique_values = dataset_preview[group_field].nunique()
    if unique_values <= 1:
        logger.warning(f"Column '{group_field}' has only {unique_values} unique values. Results may not be meaningful.")
    logger.debug(f"Grouping column '{group_field}' has {unique_values} unique values.")
    
    # Build the aggregation pipeline
    pipeline = [
        {"$match": {group_field: {"$exists": True, "$ne": None}}},
        {"$group": {
            "_id": f"${group_field}",
            "count": {"$sum": 1}
        }}
    ]
    logger.debug(f"Generated group by pipeline: {json.dumps(pipeline, indent=2)}")

    return pipeline

# Example of mongo query from option 2
def generate_mongo_aggregation_query(collection_name, dataset_preview):
    numeric_columns = filter_columns_exclude_id(
        dataset_preview.select_dtypes(include=['int64', 'float64']).columns.tolist()
    )
    if not numeric_columns:
        raise ValueError("No numeric columns available in the dataset for group by.")

    agg_field = None
    if numeric_columns:
        random.seed()
        agg_field = random.choice(numeric_columns)  # Randomly pick a categorical column
    else:
        random.seed()
        agg_field = random.choice(dataset_preview.columns)  # Randomly pick any column if no categorical columns exist
        
    aggregation_operations = {
        "sum": {"operator": "$sum", "name": "total"},
        "avg": {"operator": "$avg", "name": "average"},
        "min": {"operator": "$min", "name": "minimum"},
        "max": {"operator": "$max", "name": "maximum"}
    }

    # Randomly pick an aggregation type
    random.seed()
    selected_aggregation = random.choice(list(aggregation_operations.keys()))
    aggregation_details = aggregation_operations[selected_aggregation]

    # Build the aggregation pipeline
    pipeline = [
        {
            "$group": {
                "_id": None,  # No grouping
                f"{aggregation_details['name']}_{agg_field}": {
                    aggregation_details["operator"]: f"${agg_field}"
                }
            }
        }
    ]
    return pipeline

# Example of mongo query from option 3
def generate_mongo_sort_query(collection_name, dataset_preview):
    numeric_columns = filter_columns_exclude_id(
        dataset_preview.select_dtypes(include=['int64', 'float64']).columns.tolist()
    )
    #sort_field = numeric_columns[0] if numeric_columns else dataset_preview.columns[0]

    if not numeric_columns:
        raise ValueError("No numeric columns available in the dataset for sort.")

    sort_field = None
    if numeric_columns:
        random.seed()
        sort_field = random.choice(numeric_columns)  # Randomly pick a categorical column
    else:
        random.seed()
        sort_field = random.choice(dataset_preview.columns)  # Randomly pick any column if no categorical columns exist

    # Build the aggregation pipeline
    pipeline = [
        {"$sort": {sort_field: 1}},  # Ascending order
        {"$limit": 100}  # Limit the result
    ]
    return pipeline

if __name__ == '__main__':
    app.run(debug=True)