import os
from dotenv import load_dotenv
# from chromadb.utils.embedding_functions import GoogleGeminiEmbeddingFunction
from google import genai
import chromadb

load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

# gemini_ef = GoogleGeminiEmbeddingFunction(
#     model_name = 'gemini-embedding-2',
#     task_type = 'RETRIEVAL_DOCUMENT',
# )

client = genai.Client(api_key=GEMINI_API_KEY)

chroma_client = chromadb.PersistentClient(
    path="vector_databases/chroma_db/db3/chroma_persist")

# Load, split, embed and upsert documents into ChromaDB
collection_name = "document_qa_collection"

collection = chroma_client.get_or_create_collection(
    name = collection_name
    # embedding_function = gemini_ef
)

def load_documents(directory_path):
    print(f"========= Loading Documents =========")
    documents = []
    for filename in os.listdir(directory_path):
        if filename.endswith(".txt"):
            with open (
                os.path.join(directory_path, filename), "r", encoding="utf-8"
            ) as file:
                documents.append(
                    {
                        "id": filename,
                        "text": file.read()
                    }
                )
    print(f"========= Loaded {len(documents)} Documents =========")
    return documents


def split_to_chunks(text, chunk_size=1000, chunk_overlap=200):
    print("========= Splitting Text into Chunks =========")
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start = end - chunk_overlap
    print(f"========= Split into {len(chunks)} Chunks =========")
    return chunks


def get_gemini_embedding(text):
    print(f"========= Getting Gemini Embedding for Text =========")
    response = client.models.embed_content(
        model="gemini-embedding-2",
        contents=text
    )
    # embeddings = response.data[0].embedding
    print(f"========= Received Embedding of Length response =========")
    print(response)
    return response.embeddings[0].values


def upsert_doc_to_chroma(chunked_documents):
    for doc in chunked_documents:
        print(f"Upserting document ID: {doc['id']} with embedding length: {len(doc['embedding'])}")
        collection.upsert(
            ids = [doc["id"]],
            documents = [doc["text"]],
            embeddings = [doc["embedding"]]
        )
        print(f"Upserted document ID: {doc['id']} successfully.")
    

def search_chroma(question, top_k=3):
    query_embedding = get_gemini_embedding(question)

    #search in ChromaDB
    result = collection.query(
        query_embeddings = [query_embedding],
        n_results = top_k
    )
    return result


def build_context(results):
    documents = results["documents"][0]
    context = "\n\n".join(documents)
    return context


def ask_gemini(question, context):
    prompt = f"""
    You are a question-answering assistant.

    Answer the question using only the information provided
    in the context below.

    If the answer cannot be found in the context, say:
    "I could not find enough information in the provided documents."

    Context:
    {context}

    Question:
    {question}

    Answer:
    """

    response = client.models.generate_content(
        model="gemini-3.8-flash",
        contents=prompt,
    )
    return response.text

    
if __name__ == "__main__":
    # chunked_documents = []
    # directory_path = "vector_databases/chroma_db/news_articles"
    # documents = load_documents(directory_path)
    # for doc in documents:
    #     chunks = split_to_chunks(doc['text'])
    #     for idx, chunk in enumerate(chunks):
    #         chunked_documents.append({
    #             "id": f"{doc['id']}_chunk_{idx}",
    #             "text": chunk
    #         })
    # for doc in chunked_documents:
    #     doc['embedding'] = get_gemini_embedding(doc['text'])

    # upsert_doc_to_chroma(chunked_documents)

    question = "Tell me something about chatgpt"
    res = search_chroma(question)
    context = build_context(res)
    answer = ask_gemini(question, context)
    print(f"Question: {question}")
    print(f"Answer: {answer}")

