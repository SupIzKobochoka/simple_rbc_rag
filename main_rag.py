from langchain.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import OpenSearchVectorSearch
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from API_KEY import API_KEY_LLM

opensearch_paramns = {
    'opensearch_url': 'https://localhost:9200/',
    'http_auth': ("admin", "NEWS2025password"),
    'verify_certs': False,
    'index_name': 'langchain_test'}

model_name = 'nomic-ai/nomic-embed-text-v2-moe'

embedder = HuggingFaceEmbeddings(
    model_name=model_name,
    model_kwargs={"device": "cuda", 'trust_remote_code': True},
    encode_kwargs={'prompt_name': 'query'}
)

prompt = ChatPromptTemplate.from_template(
    "Ответь на вопрос, используя предоставленный контекст.\n\n"
    "Если в ответе используется информация из конкретной новости,\n"
    "указывай источник прямо в тексте ответа в скобках в формате:\n"
    "(дата публикации: <publish_date>, ссылка: <URL>).\n\n"
    "Если контекст не даёт полезной информации для ответа, скажи об этом явно.\n\n"
    "Сразу отвечай на вопрос, без приветственных и других слов."
    "Ответ строй в формате markdown."
    "Не забывай про ссылки и даты."
    "Пиши коротко."
    "Контекст:\n{context}\n\n"
    "Вопрос:\n{question}"
)

docsearch = OpenSearchVectorSearch(**opensearch_paramns,
                                   embedding_function=embedder,
                                   )

filter_new = {
        "range": {
            "metadata.publish_date": {
                "gte": "2017-01-01"}}}

filter_old = {
        "range": {
            "metadata.publish_date": {
                "lte": "2017-01-01"}}}

llm = ChatOpenAI(base_url="https://api.xiaomimimo.com/v1", model="mimo-v2-flash", api_key=API_KEY_LLM)

def get_rag_responce(question: str) -> str:
    context_new = docsearch.similarity_search(question, k=5, filter=filter_new)
    context_old = docsearch.similarity_search(question, k=5, filter=filter_old)

    preproc_doc = lambda doc: f'Новость: {doc.page_content}, дата публикации: {doc.metadata["publish_date"]}, URL: {doc.metadata["url"]}.'
    context = '\n'.join([f'#{ind+1} {preproc_doc(doc)}' for ind, doc in enumerate(context_new + context_old)])

    massage = prompt.invoke({'context': context, 'question': question})
    responce = llm.invoke(massage)
    return responce.content