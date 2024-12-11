from langchain_community.document_loaders import TextLoader, PyPDFLoader



# -------------載入文件----------------
pdfloader = PyPDFLoader(file_path='./pdf/船員職涯手冊.pdf')
pdf = pdfloader.load()
# print(pdf[1])


# print(pages[0].page_content)

from langchain_ollama import ChatOllama
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_text_splitters import(CharacterTextSplitter, RecursiveCharacterTextSplitter)
from langchain_google_genai import ChatGoogleGenerativeAI
import google.generativeai as genai
from langchain.chains.summarize import load_summarize_chain

def Initialize_LLM():
    # -------------分割文件-----------------
    embeddings_model = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={'device': 'cpu'},
        encode_kwargs={'normalize_embeddings': False}
    )
    chatmodel = ChatOllama(
        model = "llama3.3",
        temperature = 0.8,
        num_predict = 1024
    )
    genai.configure(api_key="GOOGLE_API_KEY")
    gemini = ChatGoogleGenerativeAI(
        model="gemini-1.5-flash",
        temperature=0.8,
    )
    # index = VectorstoreIndexCreator(embedding=embeddings_model).from_loaders([loader])
    text_splitter = RecursiveCharacterTextSplitter(
        separators=' \n',
        chunk_size=1000,
        chunk_overlap=500
    )
    chunks = text_splitter.split_documents(pdf)
    # print(chunks)
    # -----------------------文字轉向量---------------------------

    Chroma.from_documents(
        documents=chunks,
        embedding=embeddings_model,
        persist_directory='./chroma_db/db',
        collection_metadata={"hnsw:space":"cosine"}
    )
    db = Chroma(
        persist_directory='./chroma_db/db',
        embedding_function=embeddings_model
    )
    retriever = db.as_retriever(search_type="similarity",
                            search_kwargs={"k": 10})
    return chatmodel,retriever,gemini

from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_core.prompts import ChatPromptTemplate



def chatLLM(UserQuestiion,chatmodel,retriever):

    str_parser = StrOutputParser()
    template = (
        "你是一位善用工具的小幫手, 且只會回答文件相關的內容, \n"
        "請自己判斷上下文來回答問題, 不要盲目地使用工具"
        "{context}\n"
        "問題: {question}"
        )
    prompt = ChatPromptTemplate.from_template(template)

    chain = (
        {"context": retriever, "question": RunnablePassthrough()}
        | prompt
        | chatmodel
        | str_parser
    )
    llmAnwser = chain.invoke(UserQuestiion)
    print(retriever.invoke(UserQuestiion))
    return llmAnwser

