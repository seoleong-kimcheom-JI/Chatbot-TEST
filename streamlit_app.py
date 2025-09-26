import os
import tempfile
import streamlit as st

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_community.utilities import SerpAPIWrapper

from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.tools.retriever import create_retriever_tool
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.agents import create_tool_calling_agent, AgentExecutor, Tool

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

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
    return Tool(name="web_search", func=run_with_source,
                description="실시간 뉴스 및 웹 정보를 검색합니다.")

def load_pdf_files(uploaded_files):
    all_docu_
