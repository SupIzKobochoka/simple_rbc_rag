from langchain.schema import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import OpenSearchVectorSearch
import pandas as pd

data = pd.read_csv('data\data.csv')
data = data.to_dict('list')

model_name = 'nomic-ai/nomic-embed-text-v2-moe'
opensearch_paramns = {
    'opensearch_url': 'https://localhost:9200/',
    'http_auth': ("admin", "NEWS2025password"),
    'engine': "faiss",
    'verify_certs': False,
    'index_name': 'langchain_test',
    'bulk_size': 100_000}

texts = [Document(page_content=data['text'][ind], 
                  metadata={'publish_date': data['publish_date'][ind],
                            'url': data['fronturl'][ind],}
                  ) for ind in range(len(data['text']))
                  ]

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200,
    add_start_index=True
)

texts = text_splitter.split_documents(texts)

embedder = HuggingFaceEmbeddings(
    model_name=model_name,
    model_kwargs={"device": "cuda", 'trust_remote_code': True},
    encode_kwargs={'prompt_name': 'passage'}
)

OpenSearchVectorSearch.from_documents(
    texts, embedder,
    **opensearch_paramns
)