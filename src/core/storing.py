from pymilvus import MilvusClient,connections,db,utility,FieldSchema,DataType,Collection,CollectionSchema
from embeding import embed_model
from chunking import file_chunking
from text_extraction import text_extraction 
from generation import generation_ans
class Milvus:
    def Milvus_connection_create_collection():
        conn=connections.connect(host='localhost',port=19530)
        db.create_database('testdb')
        db.using_database('testdb')

        #adding fields
        id_field=FieldSchema(name='id',dtype=DataType.INT64,is_primary=True,auto_id=True)
        embedding_field=FieldSchema(name='embeddings',dtype=DataType.FLOAT_VECTOR,dim=384)
        content_field=FieldSchema(name='content',dtype=DataType.VARCHAR,max_length=2048)

        #creating schema
        schema=CollectionSchema(fields=[id_field,embedding_field,content_field])

        #creating collection
        collection=Collection(name='firstcollection',schema=schema)

        #index
        index_params={
        'metric_type':'L2',
        'index_type':'HNSW',
        'params':{
            'M':16,
            'efConstruction':200
            }
        }
        collection.create_index(field_name='embeddings',index_params=index_params)
        collection.load()
        return collection.name
    
    collection_name=Milvus_connection_create_collection()
    print("collection name created: ",collection_name)

    def connect_milvus_load_collection(dbname:str='testdb',collection_name:str='firstcollection'):
        conn=connections.connect(host='localhost',port=19530)
        db.using_database(dbname)
        collection=Collection(name=collection_name)
        collection.load()
        return collection
    
    collection=connect_milvus_load_collection()

    #inserting data into collection
    content=text_extraction.extract_text_from_pdf("./Data/sciencerag.pdf")
    text_chunks=file_chunking.recursive_chunk(text=content)
    content_embeddings=embed_model.sentence_transformer_embed(text=text_chunks,model_name='all-MiniLM-L6-v2')

    collection.insert([content_embeddings,text_chunks])      

    
    #search function in milvus for user query to retrieve relevant documents
    def search_milvus(query,collection,limit=3):
        query_embedding=embed_model.sentence_transformer_embed(text=query,model_name='all-MiniLM-L6-v2')
        search_params={
        'metric_type':'L2',
        'index_type':'HNSW',
        'params':{
            'M':16,
            'efConstruction':200
            }
        }
        results=collection.search(
            data=[query_embedding],
            anns_field='embeddings',
            param=search_params,
            limit=limit,
            output_fields=['content']
        )
        return results
    Milvus_content_retrieval=search_milvus("what is photosynthesis",collection,3)

    #filter function to get the content alone form retrieved documents
    def filter_Milvus_content_retrieval(Milvus_content_retrieval=Milvus_content_retrieval):
        contents=[]
        for result in Milvus_content_retrieval[0]:
            content=result.entity.get('content')
            contents.append(content)
        return contents
    
    filtered_Milvus_content_retrieval=filter_Milvus_content_retrieval(Milvus_content_retrieval)
    

    user_query="what is photosynthesis"
    prompt="You are a helpful assistant. Answer clearly."

    #give retrieval doc to llm code:
    response=generation_ans.generate_with_groq(filtered_Milvus_content_retrieval, user_query, prompt, model="llama-3.3-70b-versatile")
    print(response)

    ###output

    '''
    Photosynthesis is a biological process where light energy is converted into chemical energy in the form of glucose (a sugar) and oxygen. 
    This process occurs in plants, algae, and some bacteria, and is vital for the survival of life on Earth as it provides the base for the food chain and releases oxygen into the atmosphere.
    '''