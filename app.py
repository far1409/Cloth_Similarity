import logging
from flask import Flask, request, jsonify
import pandas as pd
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

app = Flask(__name__)

# Set up logging
logging.basicConfig(filename='cloth_similarity.log', level=logging.INFO)

# Load and preprocess the CSV data
data = pd.read_csv('final.csv')

# Preprocessing function
def preprocess_text(text):
    """
    Preprocesses the input text by removing special characters and lowercasing it.
    
    Args:
        text (str): The input text to preprocess.
    
    Returns:
        str: The preprocessed text.
    """
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
    text = text.lower()
    return text

# Preprocess the relevant columns
data['product-discountedPrice'] = data['product-discountedPrice'].astype(str)
data['product-sizeInventoryPresent'] = data['product-sizeInventoryPresent'].astype(str)
data['product-ratingsContainer'] = data['product-ratingsContainer'].astype(str)

data['cleaned_brand'] = data['product-brand'].apply(preprocess_text)
data['cleaned_discountedPrice'] = data['product-discountedPrice'].apply(preprocess_text)
data['cleaned_sizeInventoryPresent'] = data['product-sizeInventoryPresent'].apply(preprocess_text)
data['cleaned_category'] = data['product-category'].apply(preprocess_text)
data['cleaned_ratingsContainer'] = data['product-ratingsContainer'].apply(preprocess_text)

data['consolidated_text'] = data['cleaned_brand'] + ' ' + data['cleaned_discountedPrice'] + ' ' + \
                            data['cleaned_sizeInventoryPresent'] + ' ' + data['cleaned_category'] + ' ' + \
                            data['cleaned_ratingsContainer']

# Extract useful features from consolidated text descriptions
vectorizer = TfidfVectorizer()
features = vectorizer.fit_transform(data['consolidated_text'])

def compute_similarity(input_text, vectorizer, features):
    """
    Computes the cosine similarity between the input text and the data using TF-IDF features.
    
    Args:
        input_text (str): The input text to compare with the data.
        vectorizer (TfidfVectorizer): The vectorizer used to transform the text into TF-IDF features.
        features (ndarray): The TF-IDF features of the data.
    
    Returns:
        ndarray: The similarity scores between the input text and the data.
    """
    preprocessed_input = preprocess_text(input_text)
    input_vector = vectorizer.transform([preprocessed_input])
    similarity_scores = cosine_similarity(input_vector, features)
    return similarity_scores

def find_similar_items(input_text, data, vectorizer, features, top_n=5):
    """
    Finds the most similar items to the input text based on cosine similarity and product ratings.
    
    Args:
        input_text (str): The input text to find similar items for.
        data (DataFrame): The DataFrame containing the product data.
        vectorizer (TfidfVectorizer): The vectorizer used to transform the text into TF-IDF features.
        features (ndarray): The TF-IDF features of the data.
        top_n (int, optional): The number of top similar items to retrieve. Defaults to 5.
    
    Returns:
        list: A list of dictionaries containing the product link and rating of the similar items.
    """
    similarity_scores = compute_similarity(input_text, vectorizer, features)
    top_indices = similarity_scores.argsort()[0][::-1][:top_n]
    similar_items = data.loc[top_indices, ['product-base href']].values.tolist()
    return similar_items

# Flask API endpoint for similarity search
@app.route('/similar-items', methods=['POST'])
def similar_items():
    """
    Flask API endpoint to find similar items based on the input text.
    
    Request JSON Payload:
        input_text (str): The input text to find similar items for.
        top_n (int, optional): The number of top similar items to retrieve. Defaults to 5.
    
    Returns:
        list: A list of dictionaries containing the product link and rating of the similar items.
    """
    try:
        input_text = request.json['input_text']
        top_n = request.json.get('top_n', 5)
    
        similar_items = find_similar_items(input_text, data, vectorizer, features, top_n)
        similar_items_formatted = []
        for item in similar_items:
            similar_items_formatted.append({
                'product_link': item[0]
            })

        # Log the API request and response
        logging.info(f"API Request: input_text={input_text}, top_n={top_n}")
        logging.info(f"API Response: {similar_items_formatted}")

        return jsonify(similar_items_formatted)
    except Exception as e:
        logging.error(f"API Error: {str(e)}")
        return jsonify({'error': 'An error occurred.'}), 500


if __name__ == "__main__":
    try:
        PORT = 5001
        HOST_URL ="http://"
        HOST_IP = "127.0.0.1"
        HOST_URL = HOST_URL + str(HOST_IP) + ":" + str(PORT)
        app.run(host=HOST_IP, port=PORT , debug=True)

    except Exception as ex:
        logging.error("Error Invoking server :%s" ,ex)
