import os
import streamlit as st
import tempfile

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_community.utilities import SerpAPIWrapper

from langchain_openai import OpenAIEmbeddings, ChatOpenAI

from langchain.tools.retriever import create_retriever_tool
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

from langchain.agents import create_tool_calling_agent, AgentExecutor
from langchain.agents import Tool


# .env 파일 로드(충돌 방지용)
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


# SerpAPI 검색 툴
def search_web():
    search = SerpAPIWrapper()

    def run_with_source(query: str) -> str:
        results = search.results(query)
        organic = results.get("organic_results", []) if isinstance(results, dict) else []
        formatted = []
        for r in organic[:5]:
            title = r.get("title")
            link = r.get("link")
            source = r.get("source") or r.get("displayed_link") or ""
            snippet = r.get("snippet") or ""
            if link:
                formatted.append(f"- [{title}]({link}) ({source})\n  {snippet}")
            else:
                formatted.append(f"- {title} (출처: {source})\n  {snippet}")
        return "\n".join(formatted) if formatted else "검색 결과가 없습니다."

    return Tool(
        name="web_search",
        func=run_with_source,
        description="실시간 뉴스 및 웹 정보를 검색할 때 사용합니다. 결과는 제목+출처+링크+간단요약(snippet) 형태로 반환됩니다."
    )


# PDF 업로드 → 벡터DB → 검색 툴 생성
def load_pdf_files(uploaded_files):
    all_documents = []
    for uploaded_file in uploaded_files:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
            tmp_file.write(uploaded_file.read())
            tmp_file_path = tmp_file.name
        loader = PyPDFLoader(tmp_file_path)
        documents = loader.load()
        all_documents.extend(documents)

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    split_docs = text_splitter.split_documents(all_documents)

    vector = FAISS.from_documents(split_docs, OpenAIEmbeddings())
    retriever = vector.as_retriever()

    retriever_tool = create_retriever_tool(
        retriever,
        name="pdf_search",
        description="Use this tool to search information from the pdf document"
    )
    return retriever_tool


# Agent 대화 실행
def chat_with_agent(user_input, agent_executor, session_history):
    result = agent_executor.invoke({
        "input": user_input,
        "chat_history": session_history.messages
    })
    return result["output"]


# 세션별 히스토리 관리
def get_session_history(session_id):
    if "session_history" not in st.session_state:
        st.session_state.session_history = {}
    if session_id not in st.session_state.session_history:
        st.session_state.session_history[session_id] = ChatMessageHistory()
    return st.session_state.session_history[session_id]


# 이전 메시지 출력
def print_messages():
    for msg in st.session_state.get("messa_
