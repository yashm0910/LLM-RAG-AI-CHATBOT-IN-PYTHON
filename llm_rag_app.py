import os
from langchain.document_loaders import DirectoryLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain.vectorstores import Pinecone
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI
import streamlit as st
from dotenv import load_dotenv
from langchain.vectorstores import DeepLake
from streamlit_chat import message

def configure():
    load_dotenv()

os.environ['OPENAI_API_KEY']= os.getenv('OPENAI_API_KEY')
os.environ['ACTIVELOOP_TOKEN']= os.getenv('ACTIVELOOP_TOKEN')
os.environ['DEEPLAKE_ACCOUNT_NAME']= os.getenv('DEEPLAKE_ACCOUNT_NAME')




def doc_preprocessing():
    loader = DirectoryLoader(
        'data/',
        glob='**/*.pdf',     # only the PDFs
        show_progress=True
    )
    docs = loader.load()
    text_splitter = CharacterTextSplitter(
        chunk_size=1000, 
        chunk_overlap=0
    )
    docs_split = text_splitter.split_documents(docs)
    return docs_split

st.cache_resource
def embeddings_store():
    embeddings= OpenAIEmbeddings
    print(embeddings)
    texts= doc_preprocessing()
    db=DeepLake.from_documents(texts,embeddings,dataset_path=f"hub://iamyashmaheshwari/text_embedding")
    print(db)
    db= DeepLake(
        dataset_path = f"hub://iamyashmaheshwari/text_embedding",
        read_only =True,
        embedding_function= embeddings
    )
    return db

st.cache_resource
def search_db():
    db=embeddings_store()
    retriever = db.as_retriever()
    retriever.search_kwargs['distance_metric']= 'cos'
    retriever.search_kwargs['fetch_k']= 100
    retriever.search_kwargs['maxima_marginal_relevance']= True
    retriever.search_kwargs['k']= 10
    model = ChatOpenAI(model='gpt-3.5-turbo')
    qa = RetrievalQA.from_llm(model,retriever=retriever,return_source_documents=True)
    return qa

qa= search_db()

def display_conversation(history):
    for i in range(len(history['generated'])):
        message(history['past'][i], is_user=True,key=str(i) + "_user")
        message(history['generated'][i],key=str(i))

def main():
    st.title("LLM POWERED CHATBOT")

    user_input = st.text_area("", key="input")
    if "generated" not in st.session_state:
        st.session_state["generated"]=["T'm ready to help you"]

    if "past" not in st.session_state:
        st.session_state["past"]=["hey there!"]

    if user_input:
        output = qa({"query" :user_input})    
        st.session_state.past.append(user_input)
        response = str(output["result"])
        st.session_state.generated.append(response)

    if st.session_state["generated"]:
        display_conversation(st.session_state)

if __name__=="__main__":
    main()            

    
