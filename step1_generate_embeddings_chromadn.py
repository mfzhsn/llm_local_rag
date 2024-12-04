import requests
import chromadb

# Define the API endpoint and your API key
API_URL = 'http://10.0.1.223:1234/v1/embeddings'  # Replace with the correct URL
API_KEY = 'your_api_key'  # Replace with your actual API key

client = chromadb.Client()
collection = client.create_collection(name="docs")

def generate_embeddings(text):
    headers = {
        'Authorization': f'Bearer {API_KEY}',
        'Content-Type': 'application/json'
    }

    data = {
        'input': text
    }

    response = requests.post(API_URL, json=data, headers=headers)

    if response.status_code == 200:
        return response.json()  # Adjust based on the response format
    else:
        print(f"Error: {response.status_code}, {response.text}")
        return None

if __name__ == "__main__":
    
    text_input = "Hello, World...!!!"
    
    # Generate embeddings only once
    embeddings = generate_embeddings(text_input)
    
    if embeddings:
        # Extract the embedding vectors from the response
        embedding_vector = embeddings['data'][0]['embedding']
        print("")
        print("Input text:"+ text_input)
        print("")

        print(embeddings)
        print("")
        print("The number of vectors generated are: " + str(len(embedding_vector)) + '\n') 
        # Process each character in the text input, using the generated embeddings once
        for i, d in enumerate(text_input):
            # Add each character to the collection with the single embedding vector
            collection.add(
                ids=[str(i)],
                embeddings=embedding_vector,
                documents=[d]
            )
        print("Successfully stored embeddings in DB")
    else:
        print("Failed to generate embeddings")

