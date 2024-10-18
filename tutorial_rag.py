from langchain_community.document_loaders import WebBaseLoader
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import OpenAIEmbeddings
import os
from dotenv import load_dotenv

load_dotenv()

my_key_google = os.getenv("google_apikey")
my_key_openai = os.getenv("OPENAI_API_KEY")

llm_gemini = ChatGoogleGenerativeAI(api_key=my_key_google, model="gemini-pro")
embeddings = OpenAIEmbeddings(openai_api_key=my_key_openai)

def ask_gemini(prompt):
    AI_response = llm_gemini.invoke(prompt)
    return AI_response.content


def rag_with_url(target_url, prompt):
    loader = WebBaseLoader(target_url)

    raw_documents = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=0,
        length_function=len
    )

    splitted_documents = text_splitter.split_documents(raw_documents)

    vectorstore = FAISS.from_documents(splitted_documents, embeddings)
    retriever = vectorstore.as_retriever()

    relevant_documents = retriever.get_relevant_documents(prompt)

    context_data = ""

    for document in relevant_documents:
        context_data = context_data + " " + document.page_content


    final_prompt = f"""Şöyle bir sorum var: {prompt} .
    Bu soruyu yanıtlamak için elimizde şu bilgiler var: {context_data} .
    Bu sorunun yanıtını vermek için yalnızca sana burada verdiğim eldeki bilgileri kullan."""

    AI_response = ask_gemini(prompt=final_prompt)

    return AI_response, relevant_documents


#--------------------------------------------------------------------------------


test_url = "https://bilgisayar.aku.edu.tr/2024/09/30/staj-yapan-ogrencilerin-defter-teslimi-ve-staj-mulakati-tarihi/"
test_question = "3.sınıfların staj mülakatı ne zaman?"

AI_Response, relevant_documents = rag_with_url(target_url=test_url, prompt=test_question)

print(f"Soru: {test_question}")
print("-"*100)
print(f"AI-RAG yanıtı: {AI_Response}")





