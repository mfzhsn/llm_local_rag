import requests
import chromadb
from tabulate import tabulate


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
    text_input = [
        "Per-flow hashing uses information in a packet as an input to the hash function",
        "LSR label-only hashes the packet using the labels in the MPLS stack and the incoming port (port-id)",
        "Layer 4 load balancing to include TCP/UDP source/destination port numbers in addition to source/destination IP addresses in per-flow hashing of IP packets",
    ]
    
    # Generate embeddings once for all text inputs
    embeddings = generate_embeddings(text_input)
    
    if embeddings is not None:
        for i, d in enumerate(text_input):
            collection.add(
                ids=[str(i)],
                embeddings=embeddings['data'][i]['embedding'],
                documents=[d]
            )

    ###### Retrieving ###########

    question_prompt = "How to perform load balancing using port numbers ?"

    # Generate an embedding for the prompt and retrieve the most relevant doc
    question_emb = generate_embeddings([question_prompt])  # Pass the prompt in list form
    
    if question_emb is not None:
        query_results = collection.query(
            query_embeddings=[question_emb['data'][0]['embedding']],
            n_results=10
        )
    
    rows = []
    documents = query_results['documents'][0]
    distances = query_results['distances'][0]

    for i in range(len(documents)):
        document = documents[i]
        distance = distances[i]
        rows.append((document, distance))

    # Print the results in table format
    print("")
    print(tabulate(rows, headers=["*** Document *** ", "  Distance  "], tablefmt="grid"))
    print("")
    print("The User Question: ", question_prompt)
    print('\n'"The closest distant result:"'\n')
    print(query_results['documents'][0][0])
    print("")
