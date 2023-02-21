import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer

# Read in the CSV file
df = pd.read_csv('stocks.csv')

# Combine Name and Sector columns into a single text feature
df['text'] = df['Name'] + ' ' + df['Sector']

# Create a TfidfVectorizer to convert the text features into a sparse matrix of TF-IDF weights.
tfidf = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf.fit_transform(df['text'])


# Define a function that takes user input and generates recommendations
def get_recommendations():
    # Prompt user to enter 2-5 stock symbols
    symbols = []
    while len(symbols) < 2 or len(symbols) > 5:
        symbols = input("Enter 2-5 stock symbols separated by commas: ").upper().split(',')
        symbols = [s.strip() for s in symbols]

    # Validate user input and find the rows in the DataFrame corresponding to the input symbols
    for s in symbols:
        if s not in df['Symbol'].values:
            print(f"Invalid symbol: {s}")
            return
    indices = [df.index[df['Symbol'] == s][0] for s in symbols]

    # Calculate cosine similarity between the input stocks and all other stocks
    similarities = cosine_similarity(tfidf_matrix[indices], tfidf_matrix)

    # Find the top 5 most similar stocks
    similar_indices = similarities.argsort()[0][-6:-1][::-1]

    # Return the Symbol and Name of the top 5 similar stocks
    recs = df.loc[similar_indices, ['Symbol', 'Name']]
    print(recs)


# Call the function to generate recommendations based on user input
get_recommendations()
