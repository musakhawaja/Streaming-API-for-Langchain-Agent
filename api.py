from langchain_pinecone import PineconeVectorStore
from langchain_openai import OpenAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.prompts import MessagesPlaceholder
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
from pinecone import Pinecone, ServerlessSpec
from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain.tools.retriever import create_retriever_tool
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import EmbeddingsFilter
from fastapi import FastAPI
from typing import Any
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from pydantic import BaseModel
from typing import Any
from pymongo.mongo_client import MongoClient
from pymongo.server_api import ServerApi
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
import os

load_dotenv()

uri = os.getenv("MONGO_URI")
client = MongoClient(uri, server_api=ServerApi('1'))
db = client[os.getenv("MONGO_DB")]
collection = db[os.getenv("MONGO_COL")]


pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
spec = ServerlessSpec(cloud='aws', region='us-east-1')


app = FastAPI(
    title="LangChain Server",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["*"],
)

class Input(BaseModel):
    input: str 
    session_id: str

class Output(BaseModel):
    output: Any

def get_chat_history(session_id: str):
    session_data = collection.find_one({"session_id": session_id})
    if session_data:
        return session_data["chat_history"]
    else:
        return []

def save_chat_history(session_id: str, chat_history: []):
    collection.update_one({"session_id": session_id}, {"$set": {"chat_history": chat_history}}, upsert=True)


def load_db(index_name="test"):
    db = PineconeVectorStore(index_name=index_name, embedding=OpenAIEmbeddings(model="text-embedding-ada-002", chunk_size=500))
    return db


def retriever():
    embeddings = OpenAIEmbeddings()
    embeddings_filter = EmbeddingsFilter(embeddings=embeddings, similarity_threshold=0.76)
    vector_db =  load_db(category="new")
    retriever =  vector_db.as_retriever(search_type="similarity_score_threshold", search_kwargs={"k": 5})
    compression_retriever = ContextualCompressionRetriever(
        base_compressor=embeddings_filter, base_retriever=retriever
    )
    tool = create_retriever_tool(
    compression_retriever,
    "tool_name",
    f"""Describe your tool here so the LLM knows when to use it and what to use it for
    """)
    return tool


def generate_response():
    llm = ChatOpenAI(temperature=1, model="gpt-4o", streaming = True, callbacks=[StreamingStdOutCallbackHandler()])
    prompt = ChatPromptTemplate.from_messages([
        ("system",f"""
        ENTER YOUR PROMPT HERE
    """,),
        MessagesPlaceholder(variable_name="chat_history"),
        ("user", "{input}"),
        MessagesPlaceholder("agent_scratchpad"),
    ])
    tool =  retriever()
    tools = [tool]
    agent = create_tool_calling_agent(llm, tools, prompt)
    agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True, return_intermediate_steps=True)

    return agent_executor

@app.post('/ask')
def custom_stream(input: Input):
    agent_executor = generate_response()
    session_id = input.session_id
    chat_history = get_chat_history(session_id)
    async def generator():
        ai=""
        async for event in agent_executor.astream_events(
            {
                "input": input.input,
                "chat_history": chat_history
            },
            version="v1",
        ):
            kind = event["event"]
            if kind == "on_chat_model_stream":
                content = event["data"]["chunk"].content
                if content:
                    ai+=content
                    yield content

        chat_history.append({"role": "user", "content": input.input})
        chat_history.append({"role": "assistant", "content": ai})
        save_chat_history(session_id=session_id, chat_history=chat_history)
    return StreamingResponse(generator(), media_type='text/event-stream')
    

