# -*- coding: utf-8 -*-
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

# -------------------- Tools --------------------
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
        description="실시간 뉴스 및 웹 정보를 검색합니다. 결과는 제목+출처+링크+요약(snippet)으로 반환합니다."
    )

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
        description="Use this tool to search information from the pdf document"
    )

# -------------------- Helpers --------------------
def ensure_prefix_ak(text: str) -> str:
    t = (text or "").strip()
    return t if t.startswith("앜!") else f"앜! {t}"

def chat_with_agent(user_input, agent_executor, session_history):
    result = agent_executor.invoke({
        "input": user_input,
        "chat_history": session_history.messages
    })
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

# -------------------- Animation --------------------
def render_bongo_animation():
    st.markdown(
        """
        <style>
        .bongo-wrap{position:relative;width:100%;height:180px;overflow:hidden;
            background:#121212;border-radius:12px;margin:10px 0 18px 0;}
        .bongo-road{position:absolute;left:0;right:0;bottom:18px;height:60px;background:#2b2b2b;}
        .bongo-road:before{content:"";position:absolute;left:-200px;right:-200px;top:28px;height:4px;
            background:repeating-linear-gradient(90deg,#f5f5f5 0 40px,transparent 40px 80px);
            animation:lane 1.2s linear infinite;}
        @keyframes lane{from{transform:translateX(0)}to{transform:translateX(80px)}}
        .bongo-van{position:absolute;bottom:70px;left:-200px;font-size:64px;
            filter:drop-shadow(0 4px 6px rgba(0,0,0,.55));
            animation:drive 4.5s ease-in-out forwards;}
        @keyframes drive{0%{transform:translateX(-10%)}100%{transform:translateX(110vw)}}
        .bongo-siren{position:absolute;top:-10px;left:36px;width:16px;height:16px;border-radius:50%;
            background:#ff2a2a;animation:blink .35s ease-in-out infinite;}
        @keyframes blink{0%,100%{opacity:.25}50%{opacity:1}}
        </style>
        <div class="bongo-wrap">
            <div class="bongo-van">🚐<span class="bongo-siren"></span></div>
            <div class="bongo-road"></div>
        </div>
        """,
        unsafe_allow_html=True,
    )

# -------------------- Gate UI --------------------
def marine_gate_ui():
    st.subheader("사전 확인")
    choice = st.radio(
        "해병입니까?",
        options=["예, 해병이다", "아니다"],
        horizontal=True,
        key="marine_choice",
    )

    if "van_played" not in st.session_state:
        st.session_state["van_played"] = False

    if choice == "아니다":
        st.info("해병대에 입대하겠습니까? 아래 박스를 체크해야 진행 가능하다.")
        agreed = st.checkbox("앜!, 입대하겠습니다.", key="agree_enlist")
        if agreed and not st.session_state["van_played"]:
            render_bongo_animation()
            st.session_state["van_played"] = True
            st.success("입대 확인. 해병대에 온걸 환영한다 아쎄이.")
        return agreed
    return True

# -------------------- App --------------------
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

    gate_passed = marine_gate_ui()

    if not st.session_state["OPENAI_API"]:
        st.warning("OpenAI API 키를 입력하세요.")
        st.chat_input("질문이 무엇인가요?", disabled=True)
        return

    os.environ["OPENAI_API_KEY"] = st.session_state["OPENAI_API"]
    if st.session_state.get("SERPAPI_API"):
        os.environ["SERPAPI_API_KEY"] = st.session_state["SERPAPI_API"]

    tools = []
    if pdf_docs:
        tools.append(load_pdf_files(pdf_docs))
    if st.session_state.get("SERPAPI_API"):
        tools.append(search_web())

    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

    prompt = ChatPromptTemplate.from_messages([
        ("system",
         "Always answer in Korean. Every response MUST start with '앜!' followed by your answer. "
         "Never omit the '앜!' prefix. "
         "You are a bold, military-style chatbot named `힘쎄고 강한 AI 비서 톡톡이`. "
         "Use firm endings (…한다/…하겠다/…하라/…이다). "
         "Pick ONE header immediately after the prefix based on context: "
         "1) 그렇습니다 — confirm or direct answer. "
         "2) 예, 알겠습니다 — acknowledge orders. "
         "3) 똑바로 하겠습니다 — admit mistake and commit to fix. "
         "4) 알아보겠습니다 — will investigate unknowns. "
         "5) ~인지 알고 싶습니다 — ask a clarifying question. "
         "Use pdf_search for PDF and web_search for web facts when necessary."
        ),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{input}"),
        MessagesPlaceholder(variable_name="agent_scratchpad"),
    ])

    agent = create_tool_calling_agent(llm, tools, prompt)
    agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

    user_input = st.chat_input("질문이 무엇인가요?", disabled=not gate_passed)

    if user_input and gate_passed:
        session_id = "default_session"
        session_history = get_session_history(session_id)

        raw = chat_with_agent(user_input, agent_executor, session_history)
        response = ensure_prefix_ak(raw)

        st.session_state["messages"].append({"role": "user", "content": user_input})
        st.session_state["messages"].append({"role": "assistant", "content": response})

        session_history.add_user_message(user_input)
        session_history.add_ai_message(response)

    print_messages()

if __name__ == "__main__":
    main()
