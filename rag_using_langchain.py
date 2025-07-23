
import os
from langchain_groq import ChatGroq
# os.environ["GROQ_API_KEY"] = "YOUR API KEY"
# os.environ["HUGGINGFACEHUB_API_TOKEN"] = "YOUR API KEY"

import streamlit as st
from youtube_transcript_api._api import YouTubeTranscriptApi
from youtube_transcript_api._errors import TranscriptsDisabled
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_huggingface import ChatHuggingFace
from langchain_huggingface import HuggingFaceEndpoint

from langchain_community.llms import HuggingFaceHub     



# """## Step 1a - Indexing (Document Ingestion)"""
st.title("Youtube Video Chatbot")



video_id = st.text_input("Enter the video ID") # only the ID, not full URL
try:
    # If you don’t care which language, this returns the “best” one
    transcript_list = YouTubeTranscriptApi.get_transcript(video_id)

    # Flatten it to plain text
    transcript = " ".join(chunk["text"] for chunk in transcript_list)
    

except TranscriptsDisabled:
    print("No captions available for this video.")



# """## Step 1b - Indexing (Text Splitting)"""
with st.spinner("Splitting the transcript into chunks"):
    
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = splitter.create_documents([transcript])



# """## Step 1c & 1d - Indexing (Embedding Generation and Storing in Vector Store)"""
with st.spinner("Embedding the chunks"):
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vector_store = FAISS.from_documents(chunks, embeddings)





# """## Step 2 - Retrieval"""
with st.spinner("Retrieving the chunks"):

    retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 4})





# """## Step 3 - Augmentation"""

# model = HuggingFaceEndpoint(
#     repo_id="mistralai/Mistral-7B-Instruct-v0.1",
#     task="text-generation",
#     provider="auto",  # let Hugging Face choose the best provider for you
# )

# llm = ChatHuggingFace(llm=model)
with st.spinner("Analyzing the video"):
    llm = ChatGroq(
            model="deepseek-r1-distill-llama-70b",
            max_tokens=None,
            reasoning_format="parsed",
            timeout=None,)





    # llm = HuggingFaceHub(repo_id="facebook/bart-large-cnn")



    prompt = PromptTemplate(
        template="""
        You are a helpful assistant.
        Answer ONLY from the provided transcript context.
        If the context is insufficient, just say you don't know.
        {context}
        
        Question: {question}
        """,
        input_variables = ['question']
    )


    question = st.text_input("Enter your question")

    # Ensure it's converted to string if needed








    # """## Building a Chain"""

    from langchain_core.runnables import RunnableParallel, RunnablePassthrough, RunnableLambda
    from langchain_core.output_parsers import StrOutputParser

    def format_docs(retrieved_docs):
        context_text = "\n\n".join(doc.page_content for doc in retrieved_docs)
        return context_text

    parallel_chain = RunnableParallel({
        'context': retriever | RunnableLambda(format_docs),
        'question': RunnablePassthrough()
    })


    parser = StrOutputParser()

    main_chain = parallel_chain | prompt | llm | parser

    st.write(main_chain.invoke(question))


