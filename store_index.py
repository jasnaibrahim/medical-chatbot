from src.helper import load_pdf_file,text_split,huggingface_embeddings
from pinecone.grpc import PineconeGRPC as pinecone
from langchain_pinecone import PineconeVectorStore
from dotenv import load_dotenv
from pinecone import ServerlessSpec
import os

load_dotenv()

PINECONE_API_KEY =os.environ.get('PINECONE_API_KEY')
os.environ['PINECONE_API_KEY']=PINECONE_API_KEY
HF_TOKEN=os.environ.get("HF_TOKEN")
os.environ["HF_TOKEN"]=HF_TOKEN

extracted_data=load_pdf_file(data='data/')
text_chunks=text_split(extracted_data)
embeddings=huggingface_embeddings()


pc=pinecone(api_key=PINECONE_API_KEY)

index_name="medical-chatbot"

pc.create_index(
    name=index_name,
    dimension=384,
    metric="cosine",
    spec=ServerlessSpec(
        cloud="aws",
        region="us-east-1"
    )
)

docsearch=PineconeVectorStore.from_documents(
    documents=text_chunks,
    index_name=index_name,
    embedding=embeddings
)