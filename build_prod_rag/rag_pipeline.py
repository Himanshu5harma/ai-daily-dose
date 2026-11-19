from langchain_core import Document
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma

from dotenv import load_dotenv


load_dotenv()
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")


# Sample knowledge base
KNOWLEDGE_BASE = """# LangChain Framework

LangChain is a framework for developing applications powered by language models. It was created by Harrison Chase in October 2022.

## Core Components

1. **Models**: LangChain supports various LLM providers including OpenAI, Anthropic, and local models.

2. **Prompts**: Templates for structuring inputs to language models.

3. **Chains**: Sequences of calls to models and other components.

4. **Agents**: Systems that use LLMs to determine which actions to take.

5. **Memory**: Components for persisting state between chain/agent calls.

## LangGraph

LangGraph is a library for building stateful, multi-actor applications. Key features:
- State management
- Cycles and loops
- Human-in-the-loop
- Persistence

## Pricing

LangChain itself is open source and free. LangSmith (the observability platform) has a free tier and paid plans starting at $39/month.

## Getting Started

Install with: pip install langchain langchain-openai
Create your first chain in under 10 lines of code.
"""


def create_kb():
    """Create a vector store from knowledge base."""
    

    # Split the knowledge base into chunks
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=0)
    doc = Document(page_content=KNOWLEDGE_BASE,
                   metadata={"source": "knowledge_base"})

    chunks = splitter.split_documents([doc])

    # create a vector store from the chunks

    vector_store = Chroma.from_documents(chunks, embeddings)

    return vector_store

def demo_basic_rag():
    """Demonstrate a basic RAG pipeline."""
    vector_store = create_kb()
    retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 2})
    llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)