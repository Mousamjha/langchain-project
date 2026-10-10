import os
from dotenv import load_dotenv
import chromadb.utils.embedding_functions as ef
import chromadb
load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")


default_ef = ef.GoogleGeminiEmbeddingFunction(    
    model_name= 'gemini-embedding-001',
    task_type = 'RETRIEVAL_DOCUMENT'
)

chroma_client = chromadb.PersistentClient(
    path="vector_databases/chroma_db/db2/chroma_persist",
)

collection = chroma_client.get_or_create_collection(
    name= "google-genai-embedding",
    embedding_function=default_ef
)

document = [
    {"id": "doc1", "text": "Hello, world!"},
    {"id": "doc2", "text": "How are you today?"},
    {"id": "doc3", "text": "Goodbye, see you later!"},
    {
        "id": "doc4",
        "text": "Microsoft is a technology company that develops software. It was founded by Bill Gates and Paul Allen in 1975.",
    },
    {
        "id": "doc5",
        "text": "Artificial Intelligence (AI) refers to the simulation of human intelligence in machines that are programmed to think like humans and mimic their actions.",
    },
    {
        "id": "doc6",
        "text": "Machine Learning (ML) is a subset of AI that focuses on the development of algorithms that allow computers to learn from and make predictions based on data.",
    },
    {
        "id": "doc7",
        "text": "Deep Learning is a subset of Machine Learning that uses neural networks with many layers to analyze various factors of data.",
    },
    {
        "id": "doc8",
        "text": "Natural Language Processing (NLP) is a branch of AI that helps computers understand, interpret, and respond to human language.",
    },
    {
        "id": "doc9",
        "text": "AI can be categorized into two types: Narrow AI, which is designed to perform a narrow task, and General AI, which can perform any intellectual task that a human can do.",
    },
    {
        "id": "doc10",
        "text": "Computer Vision is a field of AI that enables computers to interpret and make decisions based on visual data from the world.",
    },
    {
        "id": "doc11",
        "text": "Reinforcement Learning is an area of Machine Learning where an agent learns to make decisions by taking actions in an environment to achieve maximum cumulative reward.",
    },
    {
        "id": "doc12",
        "text": "The Turing Test, proposed by Alan Turing, is a measure of a machine's ability to exhibit intelligent behavior equivalent to, or indistinguishable from, that of a human.",
    },
]

for doc in document:
    collection.upsert(
        ids = doc["id"],
        documents = [doc["text"]]
    )

#find query

query = "The turing test"

result = collection.query(
    query_texts=[query],
    n_results = 2
)

for idx, doc in enumerate(result['documents'][0]):
    doc_id = result['ids'][0][idx]
    distance = result['distances'][0][idx]
    print(
        f" For the query: {query}, \n\n Found similar document: {doc} (ID: {doc_id}, Distance: {distance})\n\n"
    )