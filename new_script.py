import streamlit as st
import os
import json
import re
from urllib.parse import quote
from typing import TypedDict, List, Dict, Any
import requests

from langgraph.graph import StateGraph, START, END
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.utilities import WikipediaAPIWrapper, GoogleSearchAPIWrapper
from langchain_community.tools import WikipediaQueryRun, GoogleSearchRun

# ---------------------------
# 🔑 GOOGLE GENAI (DIRECT)
# ---------------------------
from google.generativeai import configure, GenerativeModel

if "GOOGLE_API_KEY" not in os.environ:
    st.error("❌ GOOGLE_API_KEY missing")
else:
    configure(api_key=os.environ["GOOGLE_API_KEY"])
    gemini = GenerativeModel("gemini-2.5-flash")

GOOGLE_CSE_ID = os.getenv("GOOGLE_CSE_ID")

# -------------------------------------------
# 📂 Vector DB Paths (relative)
# -------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

PERSIST_DIR_1 = os.path.join(BASE_DIR, "chroma_db_nomic")
PERSIST_DIR_2 = os.path.join(BASE_DIR, "chroma_db_jsonl")

# -------------------------------------------
# 🧠 Load DBs
# -------------------------------------------
embeddings = OllamaEmbeddings(model="nomic-embed-text")

db1 = Chroma(persist_directory=PERSIST_DIR_1, embedding_function=embeddings)
db2 = Chroma(persist_directory=PERSIST_DIR_2, embedding_function=embeddings)

retriever1 = db1.as_retriever(search_kwargs={"k": 8})
retriever2 = db2.as_retriever(search_kwargs={"k": 8})

# -------------------------------------------
# 🌐 External Tools
# -------------------------------------------
wiki_tool = WikipediaQueryRun(api_wrapper=WikipediaAPIWrapper())
google_tool = GoogleSearchRun(api_wrapper=GoogleSearchAPIWrapper(
    google_api_key=os.environ["GOOGLE_API_KEY"], google_cse_id=GOOGLE_CSE_ID
))

# -------------------------------------------
# 🧠 Memory Functions
# -------------------------------------------
MEMORY_FILE = "chat_memory.json"

def load_memory():
    if os.path.exists(MEMORY_FILE):
        try:
            return json.load(open(MEMORY_FILE, "r", encoding="utf-8"))
        except:
            return []
    return []

def save_memory(mem):
    json.dump(mem[-15:], open(MEMORY_FILE, "w", encoding="utf-8"), indent=2)

# -------------------------------------------
# Utility Functions
# -------------------------------------------
def clean_query(q: str) -> str:
    return re.sub(r"[\n\r]+", " ", q.strip())

def gemini_call(prompt: str) -> str:
    """Direct Gemini LLM call."""
    try:
        resp = gemini.generate_content(prompt)
        return resp.text.strip()
    except Exception as e:
        return f"[LLM Error: {e}]"

def extractive_answer(query: str, docs: List[Any]) -> str:
    ctx = "\n\n".join(f"[{i+1}] {d.page_content}" for i, d in enumerate(docs[:6]))
    prompt = f"""
Answer the question below **using only the provided CONTEXT**.
Each sentence must end with [1], [2], etc. referencing the context.
If context is insufficient, write NOINFO.

Question: {query}

CONTEXT:
{ctx}
"""
    ans = gemini_call(prompt)
    if ans.upper().startswith("NOINFO") or len(ans) < 40:
        return ""
    return ans

def scholarly_lookup(query: str, max_results=3):
    refs = []
    try:
        r = requests.get(
            f"https://api.crossref.org/works?rows={max_results}&query={quote(query)}",
            timeout=8
        ).json()

        for item in r.get("message", {}).get("items", []):
            title = item.get("title", ["Untitled"])[0]
            authors = item.get("author", [])
            author_str = ", ".join(a.get("family", "") for a in authors[:2]) or "Unknown"
            if len(authors) > 2:
                author_str += " et al."
            year = item.get("issued", {}).get("date-parts", [[None]])[0][0]
            doi = item.get("DOI", "")
            link = f"https://doi.org/{doi}" if doi else item.get("URL", "")
            refs.append(f"{author_str} ({year}). *{title}*. {link}")
        if refs:
            return refs
    except:
        pass
    return ["(No scholarly reference found)"]

def format_clickable_citations(citations: List[str]) -> str:
    out = []
    for i, c in enumerate(citations, 1):
        m = re.search(r"(https?://[^\s]+|doi\.org/[^\s)]+)", c)
        if m:
            out.append(f"[{i}] [{c}]({m.group(1)})")
        else:
            out.append(f"[{i}] {c}")
    return "\n".join(out)

# -------------------------------------------
# 🔀 Graph Workflow (nodes unchanged)
# -------------------------------------------
class GraphState(TypedDict):
    query: str
    answer: str
    context: str
    citations: List[str]
    chat_history: List[Dict[str, str]]

def db1_node(state: GraphState):
    q = clean_query(state["query"])
    docs = retriever1.get_relevant_documents(q)
    ans = extractive_answer(q, docs)
    return {**state, "context": "DB1" if ans else "", "answer": ans}

def db2_node(state: GraphState):
    q = clean_query(state["query"])
    docs = retriever2.get_relevant_documents(q)
    ans = extractive_answer(q, docs)
    return {**state, "context": "DB2" if ans else "", "answer": ans}

def google_node(state: GraphState):
    try:
        r = google_tool.run(state["query"])
        if r:
            return {**state, "context": "Google", "answer": r}
    except:
        pass
    return {**state, "context": ""}

def wiki_node(state: GraphState):
    try:
        r = wiki_tool.run(state["query"])
        if r:
            return {**state, "context": "Wikipedia", "answer": r}
    except:
        pass
    return {**state, "context": ""}

def gbif_node(state: GraphState):
    return state   # placeholder

def inat_node(state: GraphState):
    return state   # placeholder

def final_node(state: GraphState):
    if not state["answer"]:
        # fallback to generative answer
        state["answer"] = gemini_call(state["query"])
    return state

# Build workflow
workflow = StateGraph(GraphState)
workflow.add_node("db1", db1_node)
workflow.add_node("db2", db2_node)
workflow.add_node("google", google_node)
workflow.add_node("wiki", wiki_node)
workflow.add_node("gbif", gbif_node)
workflow.add_node("inat", inat_node)
workflow.add_node("final", final_node)

workflow.add_edge(START, "db1")
workflow.add_conditional_edges("db1", lambda s: s["context"], {"db1": "final", "": "db2"})
workflow.add_conditional_edges("db2", lambda s: s["context"], {"db2": "final", "": "google"})
workflow.add_conditional_edges("google", lambda s: s["context"], {"google": "final", "": "wiki"})
workflow.add_conditional_edges("wiki", lambda s: s["context"], {"wiki": "final", "": "gbif"})
workflow.add_edge("gbif", "inat")
workflow.add_edge("inat", "final")
graph = workflow.compile()

# -------------------------------------------
# 🎨 Streamlit UI
# -------------------------------------------
st.set_page_config(page_title="Hybrid RAG Chatbot", page_icon="🤖", layout="wide")
st.title("🤖 Hybrid RAG Chatbot")

if "chat" not in st.session_state:
    st.session_state.chat = load_memory()

user_input = st.text_input("Enter your question:")

if st.button("Ask"):
    if user_input.strip():
        mem = st.session_state.chat

        result = graph.invoke({
            "query": user_input,
            "answer": "",
            "context": "",
            "citations": [],
            "chat_history": mem
        })

        answer = result["answer"]
        context = result["context"]

        st.write("### 🧠 Answer")
        st.write(answer)
        st.write(f"**Source:** `{context}`")

        mem.append({"query": user_input, "answer": answer})
        save_memory(mem)
        st.session_state.chat = mem

st.write("---")
st.write("### 💬 Chat History")
for c in st.session_state.chat[-10:]:
    st.markdown(f"**You:** {c['query']}")
    st.markdown(f"**Bot:** {c['answer']}")































# import streamlit as st
# import os
# import json
# import re
# from urllib.parse import quote
# from typing import TypedDict, List, Dict, Any
# import requests

# from langgraph.graph import StateGraph, START, END
# from langchain_google_genai import ChatGoogleGenerativeAI
# from langchain_community.embeddings import OllamaEmbeddings
# from langchain_community.vectorstores import Chroma
# from langchain_community.utilities import WikipediaAPIWrapper, GoogleSearchAPIWrapper
# from langchain_community.tools import WikipediaQueryRun, GoogleSearchRun

# # -------------------------------------------
# # 🔑 API KEYS (Streamlit Secrets or OS Vars)
# # -------------------------------------------
# GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
# GOOGLE_CSE_ID = os.getenv("GOOGLE_CSE_ID")

# if not GOOGLE_API_KEY:
#     st.error("❌ GOOGLE_API_KEY is missing. Add it to your environment or Streamlit secrets.")
# if not GOOGLE_CSE_ID:
#     st.error("❌ GOOGLE_CSE_ID is missing.")

# # -------------------------------------------
# # 🤖 LLM
# # -------------------------------------------
# gemini = ChatGoogleGenerativeAI(
#     model="gemini-2.5-flash",
#     temperature=0,
#     api_key=GOOGLE_API_KEY
# )

# # -------------------------------------------
# # 📂 Vector DB Paths (relative)
# # -------------------------------------------
# BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# PERSIST_DIR_1 = os.path.join(BASE_DIR, "chroma_db_nomic")
# PERSIST_DIR_2 = os.path.join(BASE_DIR, "chroma_db_jsonl")

# # -------------------------------------------
# # 🧠 Load DBs
# # -------------------------------------------
# embeddings = OllamaEmbeddings(model="nomic-embed-text")

# db1 = Chroma(persist_directory=PERSIST_DIR_1, embedding_function=embeddings)
# db2 = Chroma(persist_directory=PERSIST_DIR_2, embedding_function=embeddings)

# retriever1 = db1.as_retriever(search_kwargs={"k": 8})
# retriever2 = db2.as_retriever(search_kwargs={"k": 8})

# # -------------------------------------------
# # 🌐 External Tools
# # -------------------------------------------
# wiki_tool = WikipediaQueryRun(api_wrapper=WikipediaAPIWrapper())
# google_tool = GoogleSearchRun(api_wrapper=GoogleSearchAPIWrapper(
#     google_api_key=GOOGLE_API_KEY, google_cse_id=Google_CSE_ID
# ))

# # -------------------------------------------
# # 🧠 Memory Functions
# # -------------------------------------------
# MEMORY_FILE = "chat_memory.json"

# def load_memory():
#     if os.path.exists(MEMORY_FILE):
#         try:
#             return json.load(open(MEMORY_FILE, "r", encoding="utf-8"))
#         except:
#             return []
#     return []

# def save_memory(mem):
#     json.dump(mem[-15:], open(MEMORY_FILE, "w", encoding="utf-8"), indent=2)

# # -------------------------------------------
# # Utility Functions (clean_query, extractive_answer, etc.)
# # -------------------------------------------
# def clean_query(q: str) -> str:
#     return re.sub(r"[\n\r]+", " ", q.strip())

# def extractive_answer(query: str, docs: List[Any]) -> str:
#     ctx = "\n\n".join(f"[{i+1}] {d.page_content}" for i, d in enumerate(docs[:6]))
#     prompt = f"""
# Answer the question below using only the provided CONTEXT.
# Each sentence should end with [1], [2], etc. referencing the numbered context.
# If context insufficient, write NOINFO.

# Question: {query}
# CONTEXT:
# {ctx}
# """
#     ans = gemini.invoke(prompt).content.strip()
#     if ans.upper().startswith("NOINFO") or len(ans) < 40:
#         return ""
#     return ans

# def scholarly_lookup(query: str, max_results=3):
#     refs = []
#     try:
#         r = requests.get(
#             f"https://api.crossref.org/works?rows={max_results}&query={quote(query)}",
#             timeout=8
#         ).json()
#         for item in r.get("message", {}).get("items", []):
#             title = item.get("title", ["Untitled"])[0]
#             authors = item.get("author", [])
#             author_str = ", ".join(a.get("family", "") for a in authors[:2]) or "Unknown"
#             if len(authors) > 2:
#                 author_str += " et al."
#             year = item.get("issued", {}).get("date-parts", [[None]])[0][0]
#             doi = item.get("DOI", "")
#             link = f"https://doi.org/{doi}" if doi else item.get("URL", "")
#             refs.append(f"{author_str} ({year}). *{title}*. {link}")
#         if refs:
#             return refs
#     except:
#         pass
#     return ["(No scholarly reference found)"]

# def format_clickable_citations(citations: List[str]) -> str:
#     out = []
#     for i, c in enumerate(citations, 1):
#         m = re.search(r"(https?://[^\s]+|doi\.org/[^\s)]+)", c)
#         if m:
#             out.append(f"[{i}] [{c}]({m.group(1)})")
#         else:
#             out.append(f"[{i}] {c}")
#     return "\n".join(out)

# # -------------------------------------------
# # 🔀 Graph Workflow
# # -------------------------------------------
# class GraphState(TypedDict):
#     query: str
#     answer: str
#     context: str
#     citations: List[str]
#     chat_history: List[Dict[str, str]]

# # Nodes are **100% the same** — reused from your script
# # (db1_node, db2_node, google_node, wiki_node, gbif_node, inat_node, final_node)

# # Build graph
# workflow = StateGraph(GraphState)
# workflow.add_node("db1", db1_node)
# workflow.add_node("db2", db2_node)
# workflow.add_node("google", google_node)
# workflow.add_node("wiki", wiki_node)
# workflow.add_node("gbif", gbif_node)
# workflow.add_node("inat", inat_node)
# workflow.add_node("final", final_node)

# workflow.add_edge(START, "db1")
# workflow.add_conditional_edges("db1", lambda s: s["context"], {"db1": "final", "no_db1": "db2"})
# workflow.add_conditional_edges("db2", lambda s: s["context"], {"db2": "final", "no_db2": "google"})
# workflow.add_conditional_edges("google", lambda s: s["context"], {"google": "final", "no_google": "wiki"})
# workflow.add_conditional_edges("wiki", lambda s: s["context"], {"wiki": "final", "no_wiki": "gbif"})
# workflow.add_conditional_edges("gbif", lambda s: s["context"], {"gbif": "final", "no_gbif": "inat"})
# workflow.add_edge("inat", "final")
# workflow.add_edge("final", END)

# graph = workflow.compile()

# # -------------------------------------------
# # 🎨 Streamlit UI
# # -------------------------------------------
# st.set_page_config(page_title="Hybrid RAG Chatbot", page_icon="🤖", layout="wide")

# st.title("🤖 Hybrid RAG Chatbot")
# st.write("Ask any question — the bot searches DB1 → DB2 → Google → Wikipedia → GBIF → iNaturalist.")

# if "chat" not in st.session_state:
#     st.session_state.chat = load_memory()

# user_input = st.text_input("Enter your question:")

# if st.button("Ask"):
#     if user_input.strip():
#         mem = st.session_state.chat

#         result = graph.invoke({
#             "query": user_input,
#             "answer": "",
#             "context": "",
#             "citations": [],
#             "chat_history": mem
#         })

#         answer = result["answer"]
#         context = result["context"]

#         st.write("### 🧠 Answer")
#         st.write(answer)
#         st.write(f"**Source:** `{context}`")

#         mem.append({"query": user_input, "answer": answer})
#         save_memory(mem)
#         st.session_state.chat = mem

# st.write("---")
# st.write("### 💬 Chat History")
# for c in st.session_state.chat[-10:]:
#     st.markdown(f"**You:** {c['query']}")
#     st.markdown(f"**Bot:** {c['answer']}")
