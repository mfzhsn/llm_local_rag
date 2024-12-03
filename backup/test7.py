import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_distances
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from renumics import spotlight
import seaborn as sns

# Sample data (example embeddings, you will replace this with actual embeddings)
def generate_embeddings(text):
    # This should be your actual embedding generation logic
    # For demonstration, we'll use random embeddings
    np.random.seed(0)
    return np.random.rand(len(text), 768)  # Random embeddings (replace with real ones)

def create_embedding_dataframe(embeddings, labels, question_index):
    # Create a DataFrame with embeddings and labels
    df = pd.DataFrame({
        'id': labels,
        'embedding': [list(e) for e in embeddings],
    })
    
    # Optionally add distances from the question (or any other metric you want to display)
    question_vector = embeddings[question_index]
    
    # Compute cosine distances from all other embeddings to the question embedding
    distances = cosine_distances([question_vector], embeddings)[0]  # From the question to all others
    
    # Add the distance column to the dataframe
    df['distance'] = np.concatenate([distances, [None]])  # Add None for the question itself
    
    return df

def plot_embeddings_with_spotlight(embeddings, labels, question_index):
    # Reduce dimensions to 3D using PCA
    pca = PCA(n_components=3)
    reduced_embeddings = pca.fit_transform(embeddings)
    
    # Apply KMeans clustering (you can adjust the number of clusters)
    kmeans = KMeans(n_clusters=3, random_state=0).fit(reduced_embeddings)
    cluster_labels = kmeans.labels_
    
    # Create the DataFrame with embeddings, labels, and distances
    df = create_embedding_dataframe(reduced_embeddings, labels, question_index)
    
    # Add the cluster label to the DataFrame
    df['cluster'] = cluster_labels
    
    # Color the points based on their cluster
    palette = sns.color_palette("Set2", len(set(cluster_labels)))
    df['color'] = df['cluster'].map(lambda x: palette[x])
    
    # Visualize the embeddings using spotlight
    spotlight.show(df, label_column='id', color_column='color', size_column='distance')

# Main script
if __name__ == "__main__":
    # Example embeddings and labels
    text_input = [
        "Per-flow hashing uses information in a packet as an input to the hash function",
        "LSR label-only hashes the packet using the labels in the MPLS stack and the incoming port (port-id)",
        "Layer 4 load balancing to include TCP/UDP source/destination port numbers in addition to source/destination IP addresses in per-flow hashing of IP packets",
    ]
    question_prompt = "How to perform load balancing using port numbers ?"

    # Generate embeddings for documents and question
    doc_embeddings = generate_embeddings(text_input)
    question_embedding = generate_embeddings([question_prompt])  # Generate for question

    if doc_embeddings is not None and question_embedding is not None:
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
