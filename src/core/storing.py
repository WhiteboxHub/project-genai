import random
import string
from pymilvus import MilvusClient

#from pymilvus import milvusclient
class Milvus:
    def __init__(self):
        self.collection_name = "my_embedding_collection"
        self.client = MilvusClient(
            uri="http://localhost:19530",
            token="root:Milvus"
        )


    def store_data(self, chunked_file, embeddings):
        # Example embeddings (assuming you have a (2127, 384) numpy array)
        # Replace this with your actual 'embeddings' variable
        # Get the number of embeddings from the input list
        num_embeddings = len(embeddings)

        # Generate unique IDs for each embedding
        def generate_random_id(length=10):
            return ''.join(random.choices(string.ascii_letters + string.digits, k=length))

        ids = [generate_random_id() for _ in range(num_embeddings)]

        # The data to insert must be a list of dictionaries, where each dictionary is an entity.
        data_to_insert = [
            {"pk": ids[i], "embedding": embeddings[i], "text": chunked_file[i]}
            for i in range(len(ids))
        ]
        print(type(data_to_insert))

        # Insert the data using the client
        insert_result = self.client.insert(
            collection_name=self.collection_name,
            data=data_to_insert
        )
        print(f"Data inserted. Insert count: {insert_result['insert_count']}")

        print(type(insert_result))

    
    def retrive_data(self, query_embedding):
        res = self.client.search(
            collection_name=self.collection_name,
            anns_field="embedding",
            data=[query_embedding],
            limit=3,
            search_params={"metric_type": "L2"},
            output_fields=["text"]
        )

        retrieved_texts = [hit.entity.get("text") for hits in res for hit in hits]
        return retrieved_texts
