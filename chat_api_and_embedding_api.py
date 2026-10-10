import importlib.metadata
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAI
from langchain_groq import ChatGroq
from google import genai
# from langchain_community.chat_models import Chat
print()
import os
load_dotenv()

def package_version(package_name: str) -> str:
    try:
        return importlib.metadata.version(package_name)
    except importlib.metadata.PackageNotFoundError:
        return f"{package_name.upper()} Not Installed"

print(f"Langchain Core version: {package_version("langchain-core")}")
print(f"Langgraph version: {package_version("langgraph")}")
print(f"Langchain OpenAI version: {package_version("langchain-openai")}")
print(f"Langchain Gemini version: {package_version("langchain-google-genai")}")

def main():
    embedding_with_gemini()



def embedding_with_gemini():
    client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
    result = client.models.embed_content(
        model="gemini-embedding-2",
        contents="What is the meaning of life"
    )
    print("Embeddings: ", result.embeddings)
    print("embedding dimension: ", len(result.embeddings[0].values))

def chat_with_gemini():    
    client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
    interaction = client.interactions.create(
        model = "gemini-3.8-flash",
        input = "Capital of France?"
    )
    print(interaction.output_text)




def chatWithGemini():
    gemini_llm = ChatGoogleGenerativeAI(
        model="gemini-3.8-flash", 
        api_key=os.getenv("GEMINI_API_KEY"),        
        )
    gemini_response = gemini_llm.invoke(
        "Write me a poem in hindi on mousam"
    )
    # print(f"OpenAI Response: {response}\n")
    print(f"Gemini Response: {gemini_response.content[0]['text']}")

if __name__ == '__main__':
    main()
