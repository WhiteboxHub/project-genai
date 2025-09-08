from pymilvus import CollectionSchema, FieldSchema, DataType, Collection,IndexType,IndexType
from pymilvus import MilvusClient


client = MilvusClient(
    uri="http://localhost:19530",
    token="root:Milvus"
)

print("Client created successfully")
print(client)

# Verify the connection and check server status
#print(f"Connection to Milvus server established: {utility.has_connection('default')}")

res = client.list_collections()

print(f"Collections: {res}")

print(type(res))





# Set the name for your new collection
collection_name = "my_embedding_collection"

client.drop_collection(
    collection_name=collection_name
)

res = client.list_collections()

print(f"Collections: {res}")

# Define the fields for the collection
# 1. A primary key field (e.g., a unique ID for each vector)
pk_field = FieldSchema(
    name="pk", 
    dtype=DataType.VARCHAR, 
    is_primary=True, 
    auto_id=False, 
    max_length=100
)

# 2. The vector field to store the embeddings
# The dim must match the dimension of your embeddings (384)
embedding_field = FieldSchema(
    name="embedding", 
    dtype=DataType.FLOAT_VECTOR, 
    dim=384
)

text_field = FieldSchema(
    name="text", 
    dtype=DataType.VARCHAR, 
    max_length=1000
)


index_params = client.prepare_index_params()
index_params.add_index(
    field_name="embedding", 
    index_type="IVF_FLAT", # Type of the index to create
    index_name="vector_index", # Name of the index to create
    metric_type="L2", # Metric type used to measure similarity
    params={
        "nlist": 64, # Number of clusters for the index
    } # Index building params

)

# Create the schema with the defined fields
schema = CollectionSchema(
    fields=[pk_field, embedding_field, text_field],
    description="Collection for storing text and embeddings"
)

# Check if the collection already exists
if collection_name in res:
    print(f"Collection '{collection_name}' already exists.")
else:
    # Use the client to create the collection
    client.create_collection(
        collection_name=collection_name,
        schema=schema,
        index_params=index_params
    )
    print(f"Collection '{collection_name}' created successfully.")




