import gradio as gr
import ollama
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings  # Updated import path
from langchain_community.vectorstores import Chroma
from langchain.document_loaders import PyMuPDFLoader  # For PDF loading

# Load the PDF data
def load_pdf(file):
    loader = PyMuPDFLoader(file.name)  # Use PyMuPDFLoader to handle PDF content
    docs = loader.load()
    return docs

# Define the function to split and store documents
def process_pdf_docs(docs):
    # Split the loaded documents into chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    splits = text_splitter.split_documents(docs)

    # Create Ollama embeddings and vector store
    embeddings = OllamaEmbeddings(model="llama3")
    vectorstore = Chroma.from_documents(documents=splits, embedding=embeddings)
    return vectorstore

# Define the function to call the Ollama Llama3 model
def ollama_llm(question, context):
    formatted_prompt = f"Question: {question}\n\nContext: {context}"
    response = ollama.chat(model='llama3', messages=[{'role': 'user', 'content': formatted_prompt}])
    return response['message']['content']

# Define the RAG setup
def rag_chain(question, vectorstore):
    retriever = vectorstore.as_retriever()
    retrieved_docs = retriever.invoke(question)
    formatted_context = "\n\n".join(doc.page_content for doc in retrieved_docs)
    return ollama_llm(question, formatted_context)

# Define the Gradio interface for PDF uploading and question answering
def get_important_facts(pdf_file, question):
    # Load and process PDF
    docs = load_pdf(pdf_file)
    vectorstore = process_pdf_docs(docs)

    # Run RAG chain
    return rag_chain(question, vectorstore)

# Create Gradio app with PDF upload and question input
iface = gr.Interface(
    fn=get_important_facts,
    inputs=[gr.File(label="Upload PDF"), gr.Textbox(lines=2, placeholder="Enter your question here...")],
    outputs="text",
    title="RAG with Llama3 on PDF",
    description="Upload a PDF and ask questions based on its content.",
)

# Launch the Gradio app
iface.launch()
