import chromadb

client = chromadb.Client()

collection_name = "my_collection"

collection = client.get_or_create_collection(name=collection_name)

documents = [
    {"id": "1", "text": "This is the first document."},
    {"id": "2", "text": "This is the second document."},
    {"id": "3", "text": "This is the third document."}]

for doc in documents:
    collection.upsert(ids=[doc["id"]], documents=[doc["text"]])

query = "first document"
results = collection.query(query_texts=[query], n_results=2)


print("Query results:")
for result in results:
    print(f"ID: {result['id']}, Text: {result['text']}")


if __name__ == "__main__":
    # Run the application
    print("Application is running...")
