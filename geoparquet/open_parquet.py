"""open examine parquet file."""

import duckdb

# Path to your Parquet file
parquet_file_path = "io-lulc-9-class.parquet"

# Initialize a DuckDB connection. Using ":memory:" to run in-memory without creating a physical database file.
conn = duckdb.connect(database=":memory:", read_only=False)

# Load the Parquet file into a DuckDB relation (temporary view)
relation = conn.from_parquet(parquet_file_path)

# Display the first 5 rows of the dataset to examine
print("First 5 rows of the dataset:")
print(relation.limit(5).df())

# Count the total number of rows in the dataset
total_rows = relation.aggregate("count(*) as total_rows").df()
print(f"\nTotal number of rows: {total_rows['total_rows'][0]}")

row_count_result = conn.execute(
    f"SELECT COUNT(*) FROM '{parquet_file_path}'"
).fetchall()
row_count = row_count_result[0][0]
print(f"\nTotal number of rows: {row_count}")

# Proceed with getting column names if the file is not empty
if row_count > 0:
    # Fetching the first row as a DataFrame to get column names
    first_row_df = relation.limit(1).df()
    columns = first_row_df.columns
    print("\nColumns:", columns)

# Close the DuckDB connection
conn.close()
