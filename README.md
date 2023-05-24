# Flask API for Clothes Similarity 
This Flask API provides a similarity search functionality for finding similar items based on an input text query. The API uses TF-IDF (Term Frequency-Inverse Document Frequency) and cosine similarity to compute the similarity between the input text and the data.
## API Endpoints
/similar-items (POST)
This endpoint performs the similarity search based on the input text query.
### Request Body Parameters
input_text (string, required): The input text query for finding similar items.
top_n (integer, optional): The number of similar items to retrieve. Defaults to 5.
### Response
The API returns a JSON response with the following format:
#### json
[
  {
    "product_link": "URL1"
]
product_link (string): The URL of the similar product.
rating (string): The rating of the similar product.
Example Usage

{
  "input_text": "POLO shirts under 500",
}' 

# url: http://127.0.0.1:5001/similar-items

## How to Run API Locally:

### Steps to follow:
1. Install the required dependencies listed in the requirements.txt file.

pip install -r requirements.txt

Run the Flask API using the following command:
python app.py
The API will start running on "http://127.0.0.1:5001"

## Logging
The API logs the request and response details to a log file named cloth_similarity.log. The log file contains information about each API request, including the input text and the retrieved similar items.

Please note that this documentation assumes you have the necessary data file (final.csv) and the required dependencies installed. Adjust the API endpoint URL and other configurations based on your deployment environment.

## Note: 
Tool Used for scraping: Instant Data Scrapper
