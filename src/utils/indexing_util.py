from pymilvus import DataType


milvus_indexes = {
    "FLAT": {
        "field_name": "embeddings",
        "metric_type": "COSINE",
        "index_type": "FLAT",
        "index_name": "vector_index",
        "params": {}
    },
    "IVF_FLAT": {
        "field_name": "embeddings",
        "metric_type": "COSINE",
        "index_type": "IVF_FLAT",
        "index_name": "vector_index",
        "params": {"nlist": 128}
    },
    "IVF_SQ8": {
        "field_name": "embeddings",
        "metric_type": "COSINE",
        "index_type": "IVF_SQ8",
        "index_name": "vector_index",
        "params": {"nlist": 128}
    },
    "IVF_PQ": {
        "field_name": "embeddings",
        "metric_type": "COSINE",
        "index_type": "IVF_PQ",
        "index_name": "vector_index",
        "params": {"nlist": 128, "m": 8}
    },
    "HNSW": {
        "field_name": "embeddings",
        "metric_type": "COSINE",
        "index_type": "HNSW",
        "index_name": "vector_index",
        "params": {"M": 16, "efConstruction": 200}
    },
    "HNSW_SQ": {
        "field_name": "embeddings",
        "metric_type": "COSINE",
        "index_type": "HNSW_SQ",
        "index_name": "vector_index",
        "params": {"M": 16, "efConstruction": 200}
    },
    "HNSW_PQ": {
        "field_name": "embeddings",
        "metric_type": "COSINE",
        "index_type": "HNSW_PQ",
        "index_name": "vector_index",
        "params": {"M": 16, "efConstruction": 200, "m": 8}
    },
    "HNSW_PRQ": {
        "field_name": "embeddings",
        "metric_type": "COSINE",
        "index_type": "HNSW_PRQ",
        "index_name": "vector_index",
        "params": {"M": 16, "efConstruction": 200}
    },
    "SCANN": {
        "field_name": "embeddings",
        "metric_type": "COSINE",
        "index_type": "SCANN",
        "index_name": "vector_index",
        "params": {"nlist": 128, "reorder_k": 100}
    },
    "GPU_IVF_FLAT": {
        "field_name": "embeddings",
        "metric_type": "COSINE",
        "index_type": "GPU_IVF_FLAT",
        "index_name": "vector_index",
        "params": {"nlist": 128}
    },
    "GPU_IVF_PQ": {
        "field_name": "embeddings",
        "metric_type": "COSINE",
        "index_type": "GPU_IVF_PQ",
        "index_name": "vector_index",
        "params": {"nlist": 128, "m": 8}
    },
    "DISKANN": {
        "field_name": "embeddings",
        "metric_type": "COSINE",
        "index_type": "DISKANN",
        "index_name": "vector_index",
        "params": {"search_list_size": 100}
    },
    "BIN_FLAT": {
        "field_name": "embeddings",
        "metric_type": "HAMMING",
        "index_type": "BIN_FLAT",
        "index_name": "vector_index",
        "params": {}
    },
    "BIN_IVF_FLAT": {
        "field_name": "embeddings",
        "metric_type": "HAMMING",
        "index_type": "BIN_IVF_FLAT",
        "index_name": "vector_index",
        "params": {"nlist": 128}
    },
    "SPARSE_INVERTED_INDEX": {
        "field_name": "embeddings",
        "metric_type": "IP",
        "index_type": "SPARSE_INVERTED_INDEX",
        "index_name": "vector_index",
        "params": {}
    },
    "SPARSE_WAND": {  # deprecated alias via inverted_index_algo
        "field_name": "embeddings",
        "metric_type": "IP",
        "index_type": "SPARSE_INVERTED_INDEX",
        "index_name": "vector_index",
        "params": {"inverted_index_algo": "DAAT_WAND"}
    }
}


index_field_data = [
    {
        'field_name': "pk",
        'datatype': DataType.VARCHAR,
        "is_primary":True,
        "max_length":100
    },{
        'field_name':'embeddings', 'datatype':DataType.FLOAT_VECTOR,'dim':384
    },{
        'field_name':'text', 'datatype':DataType.VARCHAR , 'max_length' : 800
    }
]