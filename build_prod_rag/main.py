from dotenv import load_dotenv
from langchain_core import __version__ as langchain_core_version
from langchain_groq import ChatGroq

# Alternatives:
# from langchain_openai import ChatOpenAI
# llm = ChatOpenAI(model_name="gpt-4o", temperature=0)
# llm = ChatOpenAI(
#     model="openai/gpt-oss-120b",
#     base_url="https://api.groq.com/openai/v1",
#     api_key=os.environ["GROQ_API_KEY"],
#     temperature=0,
# )


def main():
    load_dotenv()
    print(f"langchain_core version: {langchain_core_version}")
    llm = ChatGroq(model="openai/gpt-oss-120b", temperature=0)
    response = llm.invoke("Hello from build-prod-rag!")
    print(response)


if __name__ == "__main__":
    main()
