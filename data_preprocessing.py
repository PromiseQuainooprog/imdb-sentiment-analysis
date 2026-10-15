# Import the required libraries
import pandas as pd
import numpy as np
import os

# Load the original dataset
raw_data = pd.read_csv("IMDB Dataset.csv")

# Display the number of rows and columns
print("Dataset shape:", raw_data.shape) 

# Display columns names 
print("Column names:", raw_data.columns.tolist())

# Display the data types
print(raw_data.dtypes)

# Check for missing values
print(raw_data.isnull().sum())

# Check for duplicate rows
print("Duplicate rows:", raw_data.duplicated().sum())

# Examine the sentiment categories
print(raw_data["sentiment"].value_counts())

# =======================================
# STEP 2: DATA CLEANING AND PREPROCESSING
#========================================

# Create a separate copy of the raw dataset
imdb_cleaned = raw_data.copy()

# 1. Check missing values before cleaning
print("Missing values before cleaning:",imdb_cleaned.isnull().sum())

# 2. Remove rows with missing reviews and sentiment labels
imdb_cleaned = imdb_cleaned.dropna(
    subset=["review","sentiment"]
)

# 3. Remove duplicate rows
imdb_cleaned = imdb_cleaned.drop_duplicates()

# 4. Remove unnecessary spaces from text
imdb_cleaned["review"] = imdb_cleaned["review"].str.strip()

# 5. Remove unnecessary space and standerdize sentiment labels
imdb_cleaned["sentiment"] =imdb_cleaned["sentiment"].str.strip().str.lower()


# 6. Keep only valid sentiment labels
imdb_cleaned = imdb_cleaned[
    imdb_cleaned["sentiment"].isin(["positive","negative"])
].copy()

# 7. Reset the row index
imdb_cleaned = imdb_cleaned.reset_index(drop=True)

# Dispaly the result
print("Dataset shape before cleaning:", raw_data.shape)
print("Datset shape after cleaning:",imdb_cleaned.shape)

print("Missing values after cleaning:",imdb_cleaned.isnull().sum())

print("Duplicate rows after cleaning:",imdb_cleaned.duplicated().sum())

# Create the folder pathif it does not exist
os.makedirs("data/processed", exist_ok=True)

# 9. Save tye cleaned dataset separately
imdb_cleaned.to_csv(
    "data/processed/imdb_cleaned.csv",
    index=False)
print("Cleaned dataset saved successfully!")