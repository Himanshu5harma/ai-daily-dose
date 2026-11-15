import os
from dotenv import load_dotenv
load_dotenv()

from langchain_core import __version__ as langchain_core_version
# from langgraph import __version__ as langgraph_version
from langchain_openai import ChatOpenAI
from langchain_groq import ChatGroq

print(f"langchain_core version: {langchain_core_version}")
# print(f"langgraph version: {langgraph_version}")



def main():
    # llm = ChatOpenAI(model_name="gpt-4o", temperature=0)
    llm = ChatGroq(model="openai/gpt-oss-120b", temperature=0)
    # llm = ChatOpenAI(
    # model="openai/gpt-oss-120b",
    # base_url="https://api.groq.com/openai/v1",
    # api_key=os.environ["GROQ_API_KEY"],
    # temperature=0,
    # )
    response = llm.invoke("Hello from build-prod-rag!")
    print(response)
    print("Hello from build-prod-rag!")


if __name__ == "__main__":
    main()
