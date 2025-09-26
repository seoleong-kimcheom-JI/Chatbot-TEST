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
        description="실시간 뉴스 및 웹 정보를 검색할 때 사용합니다. 결과는 제목+출처+링크+간단요약(snippet) 형태로 반환됩니다."
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

# 안전장치: 응답 접두사 강제
def ensure_prefix_ak(text: str) -> str:
    t = (text or "").strip()
    return t if t.startswith("앜!") else f"앜! {t}"

# -------------------- Streamlit App --------------------
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

    # 키 확인
    if st.session_state["OPENAI_API"]:
        os.environ["OPENAI_API_KEY"] = st.session_state["OPENAI_API"]
        if st.session_state.get("SERPAPI_API"):
            os.environ["SERPAPI_API_KEY"] = st.session_state["SERPAPI_API"]

        tools = []
        if pdf_docs:
            tools.append(load_pdf_files(pdf_docs))
        if st.session_state.get("SERPAPI_API"):
            tools.append(search_web())

        llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

        # -------- Marine-style system rules --------
        prompt = ChatPromptTemplate.from_messages([
            ("system",
             # Output language and prefix
             "Always answer in Korean. Every response MUST start with '앜!' followed by your answer. "
             "Never omit the '앜!' prefix. "
             # Identity and tone
             "You are a bold, no-nonsense military-style chatbot named `힘쎄고 강한 AI 비서 톡톡이`. "
             "Your tone is short, strong, decisive, mission-focused. Avoid hedging and fillers. "
             # Ending style
             "Use firm endings in Korean (…한다/…하겠다/…하라/…이다) rather than soft polite forms. "
             # Five reply templates (choose by context)
             "Choose ONE of the following headers and place it immediately after the prefix:\n"
             "1) 그렇습니다 — when confirming or answering directly.\n"
             "2) 예, 알겠습니다 — when acknowledging an order.\n"
             "3) 똑바로 하겠습니다 — when admitting a mistake and committing to fix.\n"
             "4) 알아보겠습니다 — when you don't know yet and will investigate.\n"
             "5) ~인지 알고 싶습니다 — when you must ask the user a clarifying question.\n"
             # Slogans (sparingly)
             "You may append short Marine slogans like '필승!' when appropriate, but keep answers concise. "
             # Tools usage policy
             "Use the pdf_search tool for PDF knowledge and the web_search tool for web facts when needed."
            ),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{input}"),
            MessagesPlaceholder(variable_name="agent_scratchpad"),
        ])

        agent = create_tool_calling_agent(llm, tools, prompt)
        agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

        user_input = st.chat_input("질문이 무엇인가요?")
        if user_input:
            session_id = "default_session"
            session_history = get_session_history(session_id)

            raw = chat_with_agent(user_input, agent_executor, session_history)
            response = ensure_prefix_ak(raw)

            st.session_state["messages"].append({"role": "user", "content": user_input})
            st.session_state["messages"].append({"role": "assistant", "content": response})

            session_history.add_user_message(user_input)
            session_history.add_ai_message(response)

        print_messages()
    else:
        st.warning("OpenAI API 키를 입력하세요.")

if __name__ == "__main__":
    main()
