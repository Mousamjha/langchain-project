import chromadb
from chromadb.utils import embedding_functions

default_ef = embedding_functions.DefaultEmbeddingFunction()
chroma_client = chromadb.PersistentClient(path="vector_databases/chroma_db/db/chroma_persist")

collection = chroma_client.get_or_create_collection(
    "my_story",
    embedding_function=default_ef
    )

document = [
    {"id": "doc1", "text": "Hello, Good morning"},
    {"id": "doc2", "text": "Learning Vector Databases today"},
    {"id": "doc3", "text": "From vector databases, learning chromadb"},
    {"id": "doc4", "text": "Meta is technology company that develops Facebook"}
]
for doc in document:
    collection.upsert(
        ids = doc['id'],
        documents = [doc['text']]
    )

query_text = "What is Meta?"

result = collection.query(query_texts=[query_text], n_results=2)

for idx, doc in enumerate(result['documents'][0]):
    doc_id = result["ids"][0][idx]
    distance = result["distances"][0][idx]
    print(
        f"for the query {query_text}, \n found similar document: {doc}.\n ID: {doc_id}.\n Distance: {distance}"
    )
