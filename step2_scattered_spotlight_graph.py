import requests
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import cosine_distances
from mpl_toolkits.mplot3d import Axes3D
from renumics import spotlight 
import pandas as pd



# Define the API endpoint and your API key
API_URL = 'http://10.0.1.223:1234/v1/embeddings'  # Replace with the correct URL
API_KEY = 'your_api_key'  # Replace with your actual API key

# Function to generate embeddings
def generate_embeddings(texts):
    headers = {
        'Authorization': f'Bearer {API_KEY}',
        'Content-Type': 'application/json'
    }
    data = {'input': texts}
    
    try:
        response = requests.post(API_URL, json=data, headers=headers)
        response.raise_for_status()
        embeddings_data = response.json()
        
        # Extract embeddings
        if "data" in embeddings_data:
            return [item['embedding'] for item in embeddings_data['data']]
        else:
            print("Unexpected response format:", embeddings_data)
            return None
    except requests.exceptions.RequestException as e:
        print(f"API call failed: {e}")
        return None


def create_embedding_dataframe(embeddings, labels, question_index):
    # Create a DataFrame with embeddings and labels
    df = pd.DataFrame({
        'id': labels,
        'embedding': [list(e) for e in embeddings],
    })
    question_vector = embeddings[question_index]
    distances = cosine_distances([question_vector], embeddings[:question_index])[0]
    df['distance'] = np.concatenate([distances, [None]])  # Add distance column, None for the question itself
    
    return df

def plot_embeddings_with_spotlight(embeddings, labels, question_index):
    # Reduce dimensions to 2D or 3D using PCA (optional)
    pca = PCA(n_components=3)
    reduced_embeddings = pca.fit_transform(embeddings)
    
    # Create the DataFrame with embeddings, labels, and distances
    df = create_embedding_dataframe(reduced_embeddings, labels, question_index)
    
    # Visualize the embeddings using spotlight
    spotlight.show(df)

        
# Main script
if __name__ == "__main__":
    # Example embeddings and labels
    text_input = [
        "Per-flow hashing uses information in a packet as an input to the hash function",
        "LSR label-only hashes the packet using the labels in the MPLS stack and the incoming port (port-id)",
        "Layer 4 load balancing to include TCP/UDP source/destination port numbers in addition to source/destination IP addresses in per-flow hashing of IP packets",
        "This is Nokia 7750",
        "I like pizza",
        "I like also burgers",
        "High performance computing is the way to go",
        "GPU can proccess parallely",
        "CPU are good too but they mostly perform serial operation",
    ]
    question_prompt = "How to perform load balancing using port numbers ?"

    # Generate embeddings for documents and question
    doc_embeddings = generate_embeddings(text_input)
    question_embedding = generate_embeddings([question_prompt])  # Generate for question

    if doc_embeddings and question_embedding:
        # Combine embeddings and labels
        combined_embeddings = np.array(doc_embeddings + question_embedding)
        labels = text_input + [f"Question: {question_prompt}"]

        # Question index (last one)
        question_index = len(doc_embeddings)

        print(f"Combined Embeddings shape: {combined_embeddings.shape}")

        print("Visualizing embeddings with spotlight...")
        plot_embeddings_with_spotlight(combined_embeddings, labels, question_index)
    else:
        print("Failed to generate embeddings. Please check the API or input.")