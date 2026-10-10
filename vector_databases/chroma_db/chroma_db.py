import chromadb
from chromadb.utils import embedding_functions

chroma_client = chromadb.Client()
default_ef = embedding_functions.DefaultEmbeddingFunction()

def trial_run():
    collection_name = "test_collection"

    collection = chroma_client.get_or_create_collection(collection_name, embedding_function=default_ef)

    document = [
        {"id": "doc1", "text": "Hello, Good morning"},
        {"id": "doc2", "text": "Learning Vector Databases today"},
        {"id": "doc3", "text": "From vector databases, learning chromadb"}
    ]

    for row in document:
        collection.upsert(ids=row["id"], documents=[row["text"]])
        print(row)
    query = "Learning today?"

    result = collection.query(query_texts=[query], n_results=2)
    for idx, documents in enumerate(result["documents"][0]):
        doc_id = result["ids"][0][idx]
        distance = result["distances"][0][idx]
        print(
            f"for the query {query}, \n found similar document: {documents}.\n ID: {doc_id}.\n Distance: {distance}"
        )


if __name__ == "__main__":
    trial_run()