import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

# English example documents (Corpus)
corpus = [
    "This is the first document",
    "This document is the second document",
    "And this is the third one",
    "The first document and the second document",
]

# 1. Create, fit, and transform TfidfVectorizer object (TF-IDF Calculation)
vectorizer = TfidfVectorizer()
# Use fit_transform() to simultaneously learn and calculate the TF-IDF matrix X.
X = vectorizer.fit_transform(corpus)

# 2. Extract the list of words (Feature Names)
feature_names = vectorizer.get_feature_names_out()

# 3. Convert the TF-IDF matrix to a DataFrame and display
# X.toarray() converts the Sparse Matrix into a regular array.
tfidf_matrix = X.toarray()

# Name the documents 'Doc 1', 'Doc 2', etc.
document_names = [f"Doc {i + 1}" for i in range(len(corpus))]
df_tfidf = pd.DataFrame(tfidf_matrix, index=document_names, columns=feature_names)

print("--- Final TF-IDF Matrix (Word Importance) ---")
# Set display context to show only 4 decimal places for better readability.
with pd.option_context("display.float_format", " {:.4f}".format):
    print(df_tfidf)
print("-" * 40)

# 4. Extract IDF values and sort by sparsity (Keeping existing code)
idf_scores = vectorizer.idf_
idf_df = pd.DataFrame({"Term": feature_names, "IDF Score": idf_scores})
# Sort by IDF score in descending order; higher score means higher sparsity/rarity.
idf_df = idf_df.sort_values(by="IDF Score", ascending=False).reset_index(drop=True)

print("--- Term Sparsity Determination (Based on IDF Score) ---")
print(idf_df)
print("\n")

# 5. Example of determining important words using Sparsity (IDF) (Keeping existing code)
threshold = 1.6
print(f"--- High Sparsity Words (IDF Score > {threshold}) ---")
high_sparsity_words = idf_df[idf_df["IDF Score"] > threshold]
print(high_sparsity_words)

# TF(t, d)
# General calculation: (Count of term t in document d) / (Total number of words in document d)

# IDF(t)
# Scikit-learn's smooth idf formula = ln( (1+n) / (1+df(t) ) + 1, where n = total number of documents, df(t) = number of documents containing t

# TF-IDF(t, d) = TF(t, d) * IDF(t)
# A high TF-IDF(t, d) means that term t appears frequently in document d, but rarely appears in the overall corpus.

# d1     word count 5
# d2     word count 6
# d3     word count 6
# d4     word count 7
# TF(one, d1) = 0 / 5
# TF(one, d2) = 0 / 6
# TF(one, d3) = 1 / 6
# TF(one, d4) = 0 / 7
# n = 4, df(one) = 1
# Smooth IDF(one) = ln((1+4) / (1+1)) + 1 = ln(5/2) + 1 = 1.91629073187
