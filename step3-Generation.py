import requests
import chromadb
import openai
from tabulate import tabulate
import json


# Define the API endpoint and your API key
LLM_API_URL = 'http://10.0.1.223:1234/v1/chat/completions'
EMBEDDINGS_API_URL = 'http://10.0.1.223:1234/v1/embeddings'
API_KEY = 'your_api_key'


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
    
    response = requests.post(EMBEDDINGS_API_URL, json=data, headers=headers)
    
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
        "Users can enable Layer 4 load balancing to include TCP/UDP source/destination port numbers in addition to source/destination IP addresses in per-flow hashing of IP packets. By including the Layer 4 information, a SA/DA default hash flow can be sub-divided into multiple finer-granularity flows if the ports used between a specific SA/DA vary. Layer 4 load balancing can be enabled or disabled at the system or interface level to improve load balancing distribution by including the TCP or UDP source and destination port of the packet to the hash function. Use the following command to enable layer 4 load balancing at the system level.",
        "configure system load-balancing l4-load-balancing"
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

    question_prompt = "How to perform load balancing using port numbers in Nokia SR-7750?"
    # question_prompt = "Write me a pySROS code to extract system name of Nokia SR-7750"


    # Generate an embedding for the prompt and retrieve the most relevant doc
    question_emb = generate_embeddings([question_prompt])  # Pass the prompt in list form
    #Prininting the number embeddings of Questions and number of embeddings
    print("The number of vectors generated from the question: " + str(len(question_emb['data'][0]['embedding'])))
    
    if question_emb is not None:
        query_results = collection.query(
            query_embeddings=[question_emb['data'][0]['embedding']],
            n_results=3
        )
    
    rows = []
    documents = query_results['documents'][0]
    distances = query_results['distances'][0]

    for i in range(len(documents)):
        document = documents[i]
        distance = distances[i]
        rows.append((document, distance))
    


    ###### Generation ###########


    print("")
    print("Question:", question_prompt)
    print("")
    print("\n" + "-"*50 + "\nPrinting LLM Response\n" + "-"*50 + "\n")

    headers = {
        'Authorization': f'Bearer {API_KEY}',
        'Content-Type': 'application/json'
    }
    
    data = {
        'model': 'llama-3.2-3b-instruct',
        'messages': [
            {"role": "system", "content": str(query_results)},
            {"role": "user", "content": question_prompt},
        ],
        'temperature' : 0.2,
        'stream' : 'true'
    }
    
    llm_response = requests.post(LLM_API_URL, json=data, headers=headers)

    
    if llm_response.status_code == 200:
        # Try to parse and pretty-print JSON response
        try:
            response_json = llm_response.json()
            if 'choices' in response_json:
                for choice in response_json['choices']:
                    print("Message Content:\n")
                    print(choice['message']['content'])
            else:
                print(json.dumps(response_json, indent=4))  # Pretty print entire response
        except ValueError:
            print("Response is not in JSON format:\n")
            print(llm_response.text)
    else:
        print(f"Error: {llm_response.status_code}, {llm_response.text}")

