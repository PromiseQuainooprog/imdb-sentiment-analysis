# =============================================
# IMDB MOVIE REVIEWS: EXPLORATORY DATA ANALYSIS
# =============================================

# Import libraries
import pandas as pd  
import numpy as np   
import matplotlib.pyplot as plt   

# Load the cleaned dataset
imdb_review = pd.read_csv("data/processed/imdb_cleaned.csv")

# ----------------------------
# 1. BASIC DATASET INFROMATION
# ----------------------------

print("Dataset Shape:", imdb_review.shape)

print("First five rows:",imdb_review.head())
print("Dataset Information:", imdb_review.info())
print("Summary Statistics:", imdb_review.describe(include="all"))

# -------------------------
# 2. SENTIMENT DISTRIBUTION
# -------------------------
print("Sentiment Counts:",imdb_review["sentiment"].value_counts())
print("Sentiment Percentage:",(imdb_review["sentiment"].value_counts(normalize=True)*100).round(2))

# Visualize sentiment distribution
sentiment_counts = imdb_review["sentiment"].value_counts()

plt.figure(figsize=(8,5))
sentiment_counts.plot(kind="bar")
plt.title("Distribution of IMDB Movies Reviews")
plt.xlabel("Sentiment")
plt.ylabel("Number of Reviews")
plt.xticks(rotation=0)
plt.tight_layout()
plt.savefig("distribution of imdb movies reviews.png")

# ----------------------------------
# 3. REVIEW LENGTH ANALYSIS
# ------------------------------------

# Calculate review length in characters
imdb_review["review_length"] = imdb_review["review"].str.len()
print("Review Length Statistics:",imdb_review["review_length"].describe())

# Calculate average review length by sentiment
print("Average Review Length by Sentiment:",imdb_review.groupby("sentiment")["review_length"].mean())

# ------------------------------
# 4. REVIEW LENGTH VISUALIZATION
# ------------------------------

plt.figure(figsize=(8,5))
for sentiment in ["positive","negative"]:
    imdb_review.loc[imdb_review["sentiment"]== sentiment, "review_length"].hist(
        bins=40,
        alpha=0.5,
        label=sentiment
    )
plt.title("Review Length Distribution by Sentiment")
plt.xlabel("Review Length (Characters)")
plt.ylabel("Number of Reviews")
plt.legend()
plt.tight_layout()
plt.savefig("review length distribution by sentiment.png")

# ---------------------------
# 5. REVIEW LENGTH COMPARISON
# ----------------------------

average_lengths = imdb_review.groupby("sentiment")["review_length"].mean()
plt.figure(figsize=(8,5))
average_lengths.plot(kind="bar")
plt.title("Average Review Length by Sentiment")
plt.xlabel("Sentiment")
plt.ylabel("Average Characters")
plt.xticks(rotation=0)
plt.tight_layout()
plt.savefig("average review length by sentiment.png")

# ---------------------------
# END OF EDA
# ---------------------------

print("EDA completed successfully")

# =============================
# STEP 5. RETREVE EDA RESULTS
# =============================

# Display the total number of reviews
print("---- DATASET SUMMMARY ------")
print("Total Reviews:", len(imdb_review))

# Count postive and negative reviews
print("---- SENTIMENT DISTRIBUTION ----")
print(imdb_review["sentiment"].value_counts())

# Calculate sentiment percentages
print("---- SENTIMENT PERCENTAGES ----")
print((imdb_review["sentiment"].value_counts(normalize=True)*100).round(2))

# Display average rview length by sentiment
print("---- AVERAGE REVIEW LENGTH ----")
print(imdb_review.groupby("sentiment")["review_length"].mean().round(2))

# Display shortest and longest reviews
print("---- REVIEWS LENGTH RANGE ----")
print("Shortages reviews:", imdb_review["review_length"].min(), "characters")
print("Longest reviews:", imdb_review["review_length"].max(), "characters")


