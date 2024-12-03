import requests
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import cosine_distances
from mpl_toolkits.mplot3d import Axes3D


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

# Function to visualize embeddings in 3D
def plot_embeddings_3d_customized(embeddings, labels, question_index):
    try:
        # Reduce dimensions to 3D using PCA
        pca = PCA(n_components=3)
        reduced_embeddings = pca.fit_transform(embeddings)

        # Calculate distances from the question to each document
        question_vector = embeddings[question_index]
        distances = cosine_distances([question_vector], embeddings[:question_index])[0]

        # Create a 3D scatter plot with custom markers
        fig = plt.figure(figsize=(14, 10))
        ax = fig.add_subplot(111, projection='3d')

        for i, label in enumerate(labels):
            x, y, z = reduced_embeddings[i]
            marker = 'o' if i != question_index else '^'  # Different marker for question
            color = 'blue' if i != question_index else 'red'  # Highlight question in red
            ax.scatter(x, y, z, label=label, s=100, marker=marker, color=color)

            # Annotate distances (skip the question point)
            if i != question_index:
                ax.text(x, y, z + 0.2, f"{distances[i]:.2f}", fontsize=9, color='green', ha='center')

        # Prepare left-aligned text for distances
        distance_text = "\n".join(
            [f"{label}: {distances[i]:.2f}" for i, label in enumerate(labels[:-1])]
        )

        # Add main title
        ax.set_title("Embeddings in 3D with Distances from Question", fontsize=14, pad=20)

        # Add left-aligned text at the top of the figure
        fig.text(
            0.01, 0.95, distance_text, fontsize=10, ha='left', va='top', family="monospace"
        )

        ax.set_xlabel("Principal Component 1")
        ax.set_ylabel("Principal Component 2")
        ax.set_zlabel("Principal Component 3")

        # Adjust layout and save
        plt.tight_layout()
        plt.savefig("embeddings_3d_with_distances.png")
        print("3D plot with left-aligned distances saved as 'embeddings_3d_with_distances.png'.")
    except Exception as e:
        print(f"Error in plotting 3D embeddings with customization: {e}")



        
# Main script
if __name__ == "__main__":
    # Text inputs and question prompt
    text_input = [
        "Per-flow hashing uses information in a packet as an input to the hash function",
        "LSR label-only hashes the packet using the labels in the MPLS stack and the incoming port (port-id)",
        "Layer 4 load balancing to include TCP/UDP source/destination port numbers in addition to source/destination IP addresses in per-flow hashing of IP packets",
        "This is Nokia 7750",
        "I like pizza",
        "I like New York",
        "High performance computing is the way to go",
        "GPU can proccess parallely",
        "CPU are good too but they mostly perform serial operation"
    ]
    question_prompt = "How to perform load balancing using port numbers?"

    print("Generating embeddings...")
    doc_embeddings = generate_embeddings(text_input)
    question_embedding = generate_embeddings([question_prompt])  # Generate for question

    if doc_embeddings and question_embedding:
        # Combine embeddings and labels
        combined_embeddings = np.array(doc_embeddings + question_embedding)
        labels = text_input + [f"Question: {question_prompt}"]

        # Question index
        question_index = len(doc_embeddings)  # Last index for the question

        print(f"Combined Embeddings shape: {combined_embeddings.shape}")

        print("Visualizing embeddings in 2D with distances...")
        plot_embeddings_3d_customized(combined_embeddings, labels, question_index)
    else:
        print("Failed to generate embeddings. Please check the API or input.")