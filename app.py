from flask import Flask,render_template,jsonify,request
from src.helper import huggingface_embeddings
from langchain_pinecone import PineconeVectorStore
from langchain_core.prompts import ChatPromptTemplate,MessagesPlaceholder
from langchain.chains import create_history_aware_retriever
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEndpoint
from langchain.chains import create_retrieval_chain
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.runnables.history import RunnableWithMessageHistory

from dotenv import load_dotenv
from langchain_core.chat_history import BaseChatMessageHistory
from src.prompt import *
import os

app=Flask(__name__)
load_dotenv()

PINECONE_API_KEY =os.environ.get('PINECONE_API_KEY')
HF_TOKEN=os.environ.get('HF_TOKEN')


os.environ['PINECONE_API_KEY']=PINECONE_API_KEY
os.environ['GROQ_API_KEY']=os.getenv("GROQ_API_KEY")
os.environ["HF_TOKEN"]=HF_TOKEN
groq_api_key=os.getenv("GROQ_API_KEY")

repo_id="meta-llama/Llama-2-7b-chat-hf"

embeddings=huggingface_embeddings()

index_name="medical-chatbot"
docsearch=PineconeVectorStore.from_existing_index(
    index_name=index_name,
    embedding=embeddings

)

retriever=docsearch.as_retriever(search_type="similarity",search_kwargs={"k":3})
llm =HuggingFaceEndpoint(temperature=0.1,max_tokens=240,repo_id=repo_id,token=HF_TOKEN)
prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        ("human", "{input}"),
    ]
)

question_answer_chain = create_stuff_documents_chain(llm, prompt)
rag_chain = create_retrieval_chain(retriever, question_answer_chain)


@app.route("/")
def index():
    return render_template('chat.html')


@app.route("/get", methods=["GET", "POST"])
def chat():
    msg = request.form["msg"]
    input = msg
    print(input)
    response = rag_chain.invoke({"input": msg})
    print("Response : ", response["answer"])
    return str(response["answer"])

if __name__ == "__main__":
    app.run(host="0.0.0.0",port=8600,debug=True)