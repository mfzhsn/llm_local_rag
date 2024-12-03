import requests
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

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
def plot_embeddings_3d(embeddings, labels):
    try:
        # Reduce dimensions to 3D using PCA
        pca = PCA(n_components=3)
        reduced_embeddings = pca.fit_transform(embeddings)
        
        # Create a 3D scatter plot
        fig = plt.figure(figsize=(10, 7))
        ax = fig.add_subplot(111, projection='3d')
        
        for i, label in enumerate(labels):
            x, y, z = reduced_embeddings[i]
            ax.scatter(x, y, z, label=label, s=50)  # s controls marker size
        
        ax.set_title("Embeddings in 3D Space")
        ax.set_xlabel("Principal Component 1")
        ax.set_ylabel("Principal Component 2")
        ax.set_zlabel("Principal Component 3")
        ax.legend(loc='best', fontsize='small')

        # Show plot or save if in a headless environment
        plt.show(block=True)  # Keep the plot open
        # For headless environments:
        # plt.savefig("embeddings_3d_plot.png")
        # print("Plot saved as 'embeddings_3d_plot.png'.")
    except Exception as e:
        print(f"Error in plotting 3D embeddings: {e}")

# Main script
if __name__ == "__main__":
    text_input = [
        "Per-flow hashing uses information in a packet as an input to the hash function",
        "LSR label-only hashes the packet using the labels in the MPLS stack and the incoming port (port-id)",
        "Layer 4 load balancing to include TCP/UDP source/destination port numbers in addition to source/destination IP addresses in per-flow hashing of IP packets",
    ]
    
    print("Generating embeddings...")
    embeddings = generate_embeddings(text_input)
    plot_embeddings_3d(embeddings, text_input)
    plt.savefig("embeddings_3d_plot.png")  # Save to file
    print("3D plot saved as 'embeddings_3d_plot.png'.")
    
    if embeddings:
        embeddings = np.array(embeddings)  # Ensure it's a NumPy array
        print(f"Embeddings shape: {embeddings.shape}")  # Debugging output
        
        print("Visualizing embeddings...")
        plot_embeddings_3d(embeddings, text_input)
    else:
        print("Failed to generate embeddings. Please check the API or input.")
