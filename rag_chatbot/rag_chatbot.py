from langchain_openai import OpenAIEmbeddings  # Updated import for embeddings
from langchain_openai import ChatOpenAI  # Import ChatOpenAI for chat model support
from langchain.chains import RetrievalQA
from langchain.text_splitter import RecursiveCharacterTextSplitter
from dotenv import load_dotenv
import os
from pathlib import Path
from langchain_community.vectorstores import FAISS  # Updated import for FAISS

# Load environment variables
load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")

if not api_key:
    raise ValueError("OPENAI_API_KEY not found in environment variables.")

# Step 1: Load University Data
with open("final_dataset.txt", "r", encoding="utf-8") as f:
    university_data = f.read()

# Step 2: Split Text with Overlap
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
documents = text_splitter.split_text(university_data)

# Step 3: Generate Embeddings and Store/Load Vector Database
vector_db_path = "uni_final_db"
embeddings = OpenAIEmbeddings(model="text-embedding-ada-002", openai_api_key=api_key)

if not Path(vector_db_path).exists():
    print("Creating a new vector database...")
    vector_db = FAISS.from_texts(documents, embeddings)
    vector_db.save_local(vector_db_path)
else:
    print("Loading existing vector database...")
    vector_db = FAISS.load_local(vector_db_path, embeddings, allow_dangerous_deserialization=True)

# Step 4: Create Retrieval Chain with Chat Model
qa_chain = RetrievalQA.from_chain_type(
    llm=ChatOpenAI(
        model="gpt-3.5-turbo",  # Chat model
        openai_api_key=api_key,
        temperature=0.7,
        max_tokens=300
    ),
    retriever=vector_db.as_retriever()
)
# Step 5: Interactive Chatbot with formatted response
print("Welcome to the University Assistant Chatbot! Type 'exit' to quit.")
while True:
    query = input("You: ").strip()
    if query.lower() == "exit":
        print("Goodbye!")
        break
    try:
        response = qa_chain.invoke(query)  # Changed from run() to invoke() as recommended
        if response:
            # Format the result to improve readability
            result = response.get('result', '')
            formatted_result = result.replace('\n', '\n- ')  # Adds bullet points to new lines
            formatted_result = f"- {formatted_result}"  # Adds bullet at the start of the first line
            print(f"Chatbot: {formatted_result}")
        else:
            print("Chatbot: Sorry, I couldn't find relevant information.")
    except Exception as e:
        print(f"Error occurred while processing your query: {e}")