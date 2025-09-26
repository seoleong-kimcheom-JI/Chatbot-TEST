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
    all_documents = []
    for uploaded_file in uploaded_files:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
            tmp_file.write(uploaded_file.read())
            tmp_file_path = tmp_file.name
        loader = PyPDFLoader(tmp_file_path)
        documents = loader.load()
        all_documents.extend(documents)

    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    split_docs = splitter.split_documents(all_documents)
    vector = FAISS.from_documents(split_docs, OpenAIEmbeddings())
    retriever = vector.as_retriever()
    return create_retriever_tool(
        retriever,
        name="pdf_search",
        description="Use this tool to search information from the pdf document",
    )

def ensure_prefix_ak(text: str) -> str:
    t = (text or "").strip()
    return t if t.startswith("앜!") else f"앜! {t}"

def chat_with_agent(user_input, agent_executor, session_history):
    result = agent_executor.invoke({"input": user_input,
                                    "chat_history": session_history.messages})
    return result["output"]

def get_session_history(session_id):
    if "session_history" not in st.session_state:
        st.session_state.session_history = {}
    if session_id not in st.session_state.session_history:
        st.session_state.session_history[session_id] = ChatMessageHistory()
    return st.session_state.session_history[session_id]

def print_messages():
    for msg in st.session_state.get("messages", []):
        st.chat_message(msg["role"]).write(msg["content"])

# ---------------- 게이트 UI: 오도 해병 여부 ----------------
def marine_gate_ui():
    st.subheader("사전 확인")
    choice = st.radio(
        "오도 해병입니까?",
        options=["예, 오도 해병이다", "아니다"],
        horizontal=True,
        key="marine_choice",
    )

    if choice == "아니다":
        st.info("해병대에 입대하겠습니까? 아래 박스를 체크해야 진행 가능하다.")
        agreed = st.checkbox("네, 입대하겠습니다.", key="agree_enlist")
        if agreed:
            st.success("입대 확인. 질문 가능하다.")
        return agreed  # 체크해야 통과
    return True  # 오도 해병이면 바로 통과

def main():
    st.set_page_config(page_title="AI 비서", layout="wide", page_icon="🤖")

    with st.container():
        st.image("./chatbot_logo.png", use_container_width=True)
        st.markdown("---")
        st.title("안녕하십니까! RAG를 활용한 'AI 비서 톡톡이' 입니다")

    if "messages" not in st.session_state:
        st.session_state["messages"] = []

    with st.sidebar:
        st.session_state["OPENAI_API"] = st.text_input("OPENAI API 키", placeholder="Enter Your API Key", type="password")
        st.session_state["SERPAPI_API"] = st.text_input("SERPAPI_API 키", placeholder="Enter Your API Key", type="password")
        st.markdown("---")
        pdf_docs = st.file_uploader("Upload your PDF Files", accept_multiple_files=True, key="pdf_uploader")

    # 게이트: 오도 해병 여부
    gate_passed = marine_gate_ui()

    if not st.session_state["OPENAI_API"]:
        st.warning("OpenAI API 키를 입력하세요.")
        # 입력창 비활성화
        st.chat_input("질문이 무엇인가요?", disabled=True)
        return

    os.environ["OPENAI_API_KEY"] = st.session_state["OPENAI_API"]
    if st.session_state.get("SERPAPI_API"):
        os.environ["SERPAPI_API_KEY"] = st.session_state["SERPAPI_API"]

    tools = []
    if pdf_docs:
