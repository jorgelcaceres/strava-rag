import pinecone, os 
pinecone.init(api_key=os.getenv("PINECONE_API_KEY"), environment=os.getenv("PINECONE_ENV")) 
print(pinecone.list_indexes())
