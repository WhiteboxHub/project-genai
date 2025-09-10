from pymilvus import MilvusClient,MilvusException
import os
from dotenv import load_dotenv
from src.utils.logger import logger
from src.utils.indexing_util import milvus_indexes,index_field_data
load_dotenv()
import uuid

# Generate a random UUID (version 4)

class chromaDB:
    def store_data():
        pass
    def retrive_data():
        pass


class Milvus:
    
    def __init__(self):
        self.url = os.getenv("MILVUS_URL","http://localhost:19530")
        self.client = MilvusClient(self.url)
        self.db_name = os.getenv("DB_NAME")
        self.collection_name = os.getenv("COLLECTION_NAME")
    @logger
    def create_database(self,db_name):
        try:
            self.client.create_database(db_name=db_name)
            print(f"Created {db_name}")
        except Exception as e:
            raise e
    
    @logger
    def list_db(self):
        return self.client.list_databases()
    
    @logger
    def discribe_db(self, db_name):
        try:
            return self.client.describe_database(db_name=db_name)
        except Exception as e:
            raise e
    
    @logger
    def create_schema(self,Schema_details : list[dict],index_type: str ,Drop_collection : bool = False):
        if self.collection_name not in self.client.list_collections() or Drop_collection is True:
            try:
                self.client.drop_collection(
                            collection_name=self.collection_name
                        )
                schema = MilvusClient.create_schema(
                    auto_id=False,
                    enable_dynamic_field=True,
                )
                index_filed_name = "embeddings"
                
                for s in Schema_details:
                    schema.add_field(**s)
                    print(schema)
                
                self.client.create_collection(
                        collection_name=self.collection_name,
                        schema=schema,
                    )
                index_data = milvus_indexes.get(index_type)
                print(index_data)
                index_params = MilvusClient.prepare_index_params()
                index_params.add_index(
                    field_name=index_filed_name,
                    metric_type=index_data.get('metric_type'),
                    index_type=index_data.get('index_type'),
                    index_name=index_data.get("index_name"),
                    params=index_data.get("params")
                )

                self.client.create_index(
                        collection_name=self.collection_name,
                        index_params=index_params,
                        sync=False
                    )
            except Exception as e:
                raise e
        
        else:
            print("the Collection already exists. Try a differnt name or drop_collection = True")
    
    @logger
    def load_collection(self):
        self.client.load_collection(collection_name=self.collection_name)
        pass

    @logger
    def insert_data(self,text_chunks : list, chunk_embeding : list):

        data = []

        for t,e in zip(text_chunks,chunk_embeding):
            info = {
                "pk": str(uuid.uuid4()),
                "embeddings" : e ,
                "text" : t
            }
            data.append(info)
        try:
            res =  self.client.insert(
                collection_name=self.collection_name,
                data=data
            )
            print(f"inserted data {res}")
            return res
        except Exception as e:
            raise e


    @logger
    def search(self,embed_text):
        try:

            result = 
            pass
        except MilvusException as e:
            raise e


    # def store_data():
    #     pass
    # def retrive_data():
    #     pass

class PGvectordb:
    def store_data():
        pass
    def retrive_data():
        pass


class pinecodedb:
    def store_data():
        pass
    def retrive_data():
        pass