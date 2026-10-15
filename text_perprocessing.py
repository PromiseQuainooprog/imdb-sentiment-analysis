# ================================
# IMDB MOVIE REVIEWS: TEXT PREPROCESSING
# =======================================

# Import libraries
import pandas as pd  
import nltk
import re  
import ssl
ssl._create_default_https_contex = ssl._create_unverified_context

from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

# --------------------------
# 1. DOWNLOAD REQUIRED NLTK RESOURCES
# ------------------------------------

nltk.download("punkt")
nltk.download("punkt_tab")
nltk.download("stopwords")

# --------------------------------------
# 2. LOAD THE CLEANED DATASET
# -------------------------------------

imdb_nltk = pd.read_csv("data/processed/imdb_cleaned.csv")
print("Datast loaded successfully!")
print("Number of reviews:", len(imdb_nltk))

# ------------------------------
# 3. PREPARE THE STOPWORDS LIST
# --------------------------------------

# Force download the resources directly before calling them
nltk.download("punkt")
nltk.download("punkt_tab")
nltk.download("stopwords")

stop_words = set(stopwords.words("english"))#

# Manual backup if your internet or campus network completely blocks NLTK downloads:
#stop_words = {"i", "me", "my", "myself", "we", "our", "ours", "ourselves", "you", "your", "yours", "he", "him", "his", "she", "her", "it", "its", "they", "them", "what", "which", "who", "whom", "this", "that", "these", "those", "am", "is", "are", "was", "were", "be", "been", "being", "have", "has", "had", "having", "do", "does", "did", "doing", "a", "an", "the", "and", "but", "if", "or", "because", "as", "until", "while", "of", "at", "by", "for", "with", "about", "against", "between", "into", "through", "during", "before", "after", "above", "below", "to", "from", "up", "down", "in", "out", "on", "off", "over", "under", "again", "further", "then", "once", "here", "there", "when", "where", "why", "how", "all", "any", "both", "each", "few", "more", "most", "other", "some", "such", "own", "same", "so", "than", "too", "very", "s", "t", "can", "will", "just", "should", "now"}

# Preserve words that are important for sentiment
negation_words = {"no","nor","nor","never"}

stop_words = stop_words - negation_words

# -------------------------------
# 4. CREATE A TEXT PREPROCESSING FUNCTION
# -------------------------------------------

def preprocess_review(text):
    # Concert text to lowercase
    text = text.lower()
    # Remove HTML tags
    text = re.sub(r"<.*?>", "",text)
    # Tokenize the review
    tokens = word_tokenize(text)
    # Keep alphabetic words remove stopwords
    tokens = [
        word for word in tokens
        if word.isalpha() and word not in stop_words
    ]
    # Return the processed tokens
    return tokens 

# --------------------------------
# 5. APPLY PREPROCESSING
# --------------------------------

imdb_nltk["tokens"] = imdb_nltk["review"].apply(preprocess_review)

# Convert token lists ino text
imdb_nltk["processed_review"] = imdb_nltk["tokens"].apply(
    lambda tokens: " ".join(tokens)
)

# ------------------------------------
# 6. DISPLAY EXAMPLES
# ------------------------------------

print("Orginal an Processed Reviews:")
for i in range(3):
   print("Original Review:",imdb_nltk.loc[i, "review"][:300])

   print("Processed Review:",imdb_nltk.loc[i, "processed_review"][:300])

   print("-"*60)

# ----------------------------------
# 7. SAVE THE PROCESSED DATASET
# ----------------------------------

output_path = "data/processed/imdb_nip_processed.csv"

imdb_nltk.to_csv(output_path, index=False)

print("Text preprocessing completed successfully!")
print("Processed dataset saved to:", output_path)
