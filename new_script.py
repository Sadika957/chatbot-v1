import streamlit as st
import os
import json
import re
import requests
from urllib.parse import quote
from typing import TypedDict, List, Dict, Any

# LangGraph
from langgraph.graph import StateGraph, START, END

# Existing Chroma DB (LangChain community wrapper)
from langchain_community.vectorstores import Chroma

# Google Search & Wikipedia Tools
from langchain_community.utilities import WikipediaAPIWrapper, GoogleSearchAPIWrapper
from langchain_community.tools import WikipediaQueryRun, GoogleSearchRun

# Google Gemini (direct API)
from google.generativeai import configure, GenerativeModel


# ======================================================
# 🔐 LOAD API KEYS FROM STREAMLIT SECRETS
# ======================================================
GOOGLE_API_KEY = st.secrets["GOOGLE_API_KEY"]
GOOGLE_CSE_ID = st.secrets["GOOGLE_CSE_ID"]

configure(api_key=GOOGLE_API_KEY)
gemini = GenerativeModel("gemini-2.5-flash")


# ======================================================
# 📁 LOAD EXISTING CHROMA DB (NO EMBEDDINGS NEEDED)
# ======================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

PERSIST_DIR_1 = os.path.join(BASE_DIR, "chroma_db_nomic")
PERSIST_DIR_2 = os.path.join(BASE_DIR, "chroma_db_jsonl")

# vectorstore already contains embeddings → do NOT pass embedding_function
db1 = Chroma(persist_directory=PERSIST_DIR_1)
db2 = Chroma(persist_directory=PERSIST_DIR_2)

retriever1 = db1.as_retriever(search_kwargs={"k": 6})
retriever2 = db2.as_retriever(search_kwargs={"k": 6})


# ======================================================
# 🌐 SEARCH TOOLS
# ======================================================
wiki_tool = WikipediaQueryRun(api_wrapper=WikipediaAPIWrapper())

google_tool = GoogleSearchRun(
    api_wrapper=GoogleSearchAPIWrapper(
        google_api_key=GOOGLE_API_KEY,
        google_cse_id=GOOGLE_CSE_ID,
    )
)


# ======================================================
# 💾 CHAT MEMORY
# ======================================================
MEMORY_FILE = "chat_memory.json"

def load_memory():
    if os.path.exists(MEMORY_FILE):
        try:
            return json.load(open(MEMORY_FILE, "r", encoding="utf-8"))
        except:
            return []
    return []


# FIX: Normalize memory to dict format
def normalize_chat(mem):
    fixed = []
    for item in mem:
        if isinstance(item, dict):
            fixed.append(item)
        elif isinstance(item, (list, tuple)) and len(item) == 2:
            fixed.append({"query": item[0], "answer": item[1]})
        else:
            continue
    return fixed


def save_memory(mem):
    json.dump(mem[-15:], open(MEMORY_FILE, "w", encoding="utf-8"), indent=2)


# ======================================================
# 🧰 UTILITY FUNCTIONS
# ======================================================
def clean_query(q: str) -> str:
    return re.sub(r"[\n\r]+", " ", q.strip())


def ask_gemini(prompt: str) -> str:
    try:
        response = gemini.generate_content(prompt)
        return response.text
    except:
        return "Gemini API Error."


def extractive_answer(query: str, docs: List[Any]) -> str:
    if not docs:
        return ""

    ctx = "\n\n".join(f"[{i+1}] {d.page_content}" for i, d in enumerate(docs[:4]))

    prompt = f"""
Use ONLY the numbered CONTEXT below to answer.

Every sentence MUST end with a citation like [1], [2], [3].

If context is insufficient, return "NOINFO".

Question: {query}

CONTEXT:
{ctx}
"""

    ans = ask_gemini(prompt)
    if ans.startswith("NOINFO") or len(ans) < 40:
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

        return refs or ["(No scholarly reference found)"]
    except:
        return ["(No scholarly reference found)"]


# ======================================================
# 🔀 GRAPH WORKFLOW NODES
# ======================================================
class GraphState(TypedDict):
    query: str
    answer: str
    context: str
    citations: List[str]
    chat_history: List[Dict[str, str]]


def db1_node(state: GraphState) -> GraphState:
    q = clean_query(state["query"])
    docs = retriever1.invoke(q)
    ans = extractive_answer(q, docs)
    return {**state, "context": "DB1" if ans else "", "answer": ans}


def db2_node(state: GraphState) -> GraphState:
    q = clean_query(state["query"])
    docs = retriever2.invoke(q)
    ans = extractive_answer(q, docs)
    return {**state, "context": "DB2" if ans else "", "answer": ans}


def google_node(state: GraphState) -> GraphState:
    try:
        res = google_tool.invoke({"query": state["query"]})
        ans = ask_gemini(f"Summarize this Google result:\n{res}")
        return {**state, "context": "Google", "answer": ans}
    except:
        return state


def wiki_node(state: GraphState) -> GraphState:
    try:
        res = wiki_tool.invoke({"query": state["query"]})
        ans = ask_gemini(f"Summarize this Wikipedia text:\n{res}")
        return {**state, "context": "Wikipedia", "answer": ans}
    except:
        return state


def final_node(state: GraphState) -> GraphState:
    q = clean_query(state["query"])
    final_answer = state["answer"] or ask_gemini(q)
    refs = scholarly_lookup(q)

    state["citations"] = refs
    state["answer"] = f"{final_answer}\n\n**References:**\n" + "\n".join(refs)
    return state


# ======================================================
# 🔧 BUILD WORKFLOW GRAPH
# ======================================================
workflow = StateGraph(GraphState)

workflow.add_node("db1", db1_node)
workflow.add_node("db2", db2_node)
workflow.add_node("google", google_node)
workflow.add_node("wiki", wiki_node)
workflow.add_node("final", final_node)

workflow.add_edge(START, "db1")
workflow.add_conditional_edges("db1", lambda s: bool(s["answer"]), {"true": "final", "false": "db2"})
workflow.add_conditional_edges("db2", lambda s: bool(s["answer"]), {"true": "final", "false": "google"})
workflow.add_conditional_edges("google", lambda s: bool(s["answer"]), {"true": "final", "false": "wiki"})
workflow.add_edge("wiki", "final")

graph = workflow.compile()


# ======================================================
# 🎨 STREAMLIT UI
# ======================================================
st.set_page_config(page_title="Hybrid RAG Chatbot", page_icon="🤖", layout="wide")
st.title("🤖 Hybrid RAG + Google + Wikipedia Chatbot")

# Load + normalize chat history
if "chat" not in st.session_state:
    st.session_state.chat = normalize_chat(load_memory())
else:
    st.session_state.chat = normalize_chat(st.session_state.chat)

user_input = st.text_input("Ask me anything:")

if st.button("Submit"):
    if user_input.strip():
        mem = st.session_state.chat

        result = graph.invoke({
            "query": user_input,
            "answer": "",
            "context": "",
            "citations": [],
            "chat_history": mem,
        })

        st.write("### Response")
        st.write(result["answer"])
        st.write(f"**Source:** `{result['context']}`")

        # Save new memory
        mem.append({"query": user_input, "answer": result["answer"]})
        save_memory(mem)
        st.session_state.chat = mem


# Display history safely
st.write("---")
st.write("### Recent Chat History")

for c in st.session_state.chat[-10:]:
    query = c.get("query", str(c))
    answer = c.get("answer", "")
    st.markdown(f"**You:** {query}")
    st.markdown(f"**Bot:** {answer}")





































# # ============================================================
# # 🌟 HYBRID CHATBOT — STREAMLIT + GEMINI + CHROMA + LANGGRAPH
# # ============================================================

# import os
# import re
# import json
# import streamlit as st
# from typing import List, Dict, Any

# # -----------------------------
# # 🔐 Load API Keys from Streamlit Secrets
# # -----------------------------
# GOOGLE_API_KEY = st.secrets["GOOGLE_API_KEY"]
# GOOGLE_CSE_ID = st.secrets["GOOGLE_CSE_ID"]
# os.environ["GOOGLE_API_KEY"] = GOOGLE_API_KEY

# # -----------------------------
# # 🤖 Gemini LLM
# # -----------------------------
# import google.generativeai as genai
# genai.configure(api_key=GOOGLE_API_KEY)
# llm = genai.GenerativeModel("gemini-2.5-flash")

# # ============================================================
# # 🧠 Custom Gemini Embeddings (Fully Compatible)
# # ============================================================
# from google.generativeai import embed_content

# class GeminiEmbeddingFunction:
#     def __init__(self, model="models/text-embedding-004"):
#         self.model = model

#     def embed_query(self, text: str):
#         result = embed_content(model=self.model, content=text)
#         return result["embedding"]

#     def embed_documents(self, texts: List[str]):
#         return [self.embed_query(t) for t in texts]

# embeddings = GeminiEmbeddingFunction()

# # ============================================================
# # 🗂️ Chroma Vector DBs (Correct Import)
# # ============================================================
# from langchain_community.vectorstores import Chroma

# PERSIST_DIR_1 = "db1/"
# PERSIST_DIR_2 = "db2/"

# db1 = Chroma(persist_directory=PERSIST_DIR_1, embedding_function=embeddings)
# db2 = Chroma(persist_directory=PERSIST_DIR_2, embedding_function=embeddings)

# retriever1 = db1.as_retriever(search_kwargs={"k": 8})
# retriever2 = db2.as_retriever(search_kwargs={"k": 8})

# # ============================================================
# # 🌐 Tools: Google Search + Wikipedia
# # ============================================================
# from langchain_community.tools import GoogleSearchRun, WikipediaQueryRun
# from langchain_community.utilities import GoogleSearchAPIWrapper, WikipediaAPIWrapper

# google_wrapper = GoogleSearchAPIWrapper(
#     google_api_key=GOOGLE_API_KEY,
#     google_cse_id=GOOGLE_CSE_ID
# )
# google_tool = GoogleSearchRun(api_wrapper=google_wrapper)

# wiki_tool = WikipediaQueryRun(api_wrapper=WikipediaAPIWrapper())

# # ============================================================
# # 🧹 Utilities
# # ============================================================
# def clean_query(q: str) -> str:
#     return q.strip().replace("\n", " ")

# def extractive_answer(query, docs):
#     if not docs:
#         return ""

#     context = "\n\n".join([d.page_content for d in docs])

#     prompt = f"""
# Answer the question strictly from the context.

# Question: {query}

# Context:
# {context}

# If no answer is found in context, reply "" (empty).
# """

#     resp = llm.generate_content(prompt)
#     ans = resp.text.strip()

#     if ans.lower().startswith("according"):
#         ans = ans.split(":", 1)[-1].strip()

#     return ans


# # ============================================================
# # 🤖 LangGraph Hybrid Routing
# # ============================================================
# from langgraph.graph import StateGraph, START, END
# from typing import TypedDict

# class GraphState(TypedDict):
#     query: str
#     answer: str
#     context: str
#     justification: str


# # ======== DB1 Node ========
# def db1_node(state: GraphState):
#     q = clean_query(state["query"])
#     docs = retriever1.invoke(q)
#     ans = extractive_answer(q, docs)

#     return {
#         **state,
#         "context": "db1" if ans else "",
#         "answer": ans
#     }

# # ======== DB2 Node ========
# def db2_node(state: GraphState):
#     q = clean_query(state["query"])
#     docs = retriever2.invoke(q)
#     ans = extractive_answer(q, docs)

#     return {
#         **state,
#         "context": "db2" if ans else "",
#         "answer": ans
#     }

# # ======== Web Search Node ========
# def web_node(state: GraphState):
#     q = clean_query(state["query"])

#     try:
#         results = google_tool.run(q)
#         if isinstance(results, list):
#             text = "\n".join([json.dumps(r, indent=2) for r in results])
#         else:
#             text = str(results)

#         prompt = f"""
# Use only the following information to answer the question.

# Question: {q}

# Search Results:
# {text}

# Final Answer:
# """
#         resp = llm.generate_content(prompt)
#         output = resp.text.strip()

#         justify = json.dumps(results[:2], indent=2) if isinstance(results, list) else ""

#         return {**state, "context": "web", "answer": output, "justification": justify}

#     except:
#         return {**state, "context": "", "answer": ""}

# # ======== Wikipedia Node ========
# def wiki_node(state: GraphState):
#     q = clean_query(state["query"])

#     try:
#         wiki = wiki_tool.run(q)

#         prompt = f"""
# Use ONLY the following Wikipedia text to answer the question.

# Question: {q}

# Wikipedia Text:
# {wiki}

# Final Answer:
# """
#         resp = llm.generate_content(prompt)
#         output = resp.text.strip()

#         return {**state, "context": "wiki", "answer": output, "justification": wiki}

#     except:
#         return {**state, "context": "", "answer": ""}


# # ======== Router ========
# def router(state: GraphState):
#     q = state["query"].lower()

#     if any(x in q for x in ["bee", "wasp", "pollinator", "insect"]):
#         return "db1"

#     if "code" in q or "python" in q:
#         return "db2"

#     return "web"


# # ======== Build Graph ========
# graph = StateGraph(GraphState)

# graph.add_node("db1", db1_node)
# graph.add_node("db2", db2_node)
# graph.add_node("web", web_node)
# graph.add_node("wiki", wiki_node)

# graph.add_conditional_edges(
#     START,
#     router,
#     {
#         "db1": "db1",
#         "db2": "db2",
#         "web": "web"
#     }
# )

# graph.add_edge("db1", "wiki")
# graph.add_edge("db2", "wiki")
# graph.add_edge("web", END)
# graph.add_edge("wiki", END)

# graph = graph.compile()

# # ============================================================
# # 🎨 Streamlit UI
# # ============================================================

# st.set_page_config(page_title="Hybrid Chatbot", page_icon="🤖")

# st.title("🤖 Hybrid RAG Chatbot")
# st.write("Uses Gemini + Chroma + Web Search + Wikipedia + LangGraph")

# if "chat" not in st.session_state:
#     st.session_state.chat = []

# user_input = st.text_input("Ask anything:")

# if user_input.strip():
#     mem = st.session_state.chat

#     result = graph.invoke({
#         "query": user_input,
#         "answer": "",
#         "context": "",
#         "justification": ""
#     })

#     final = result["answer"]
#     source = result["context"]
#     just = result["justification"]

#     st.session_state.chat.append((user_input, final, source, just))

# # Display Chat
# for q, a, s, j in st.session_state.chat:
#     st.markdown(f"### **You:** {q}")
#     st.markdown(f"**Answer:** {a}")
#     st.markdown(f"📌 *Source Used:* `{s}`")
#     if j:
#         st.markdown(f"<details><summary>View Source</summary><pre>{j}</pre></details>",
#             unsafe_allow_html=True)
#     st.write("---")





























# # import os
# # import json
# # import re
# # from typing import TypedDict, List, Dict, Any
# # from urllib.parse import quote

# # import requests
# # import streamlit as st

# # from langgraph.graph import StateGraph, START, END
# # from langchain_community.embeddings import OllamaEmbeddings
# # from langchain_community.vectorstores import Chroma
# # from langchain_community.utilities import WikipediaAPIWrapper
# # from langchain_community.tools import WikipediaQueryRun

# # # ================================
# # # 🔑 GOOGLE GEMINI SETUP
# # # ================================
# # from google.generativeai import configure, GenerativeModel

# # GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY")

# # if not GOOGLE_API_KEY:
# #     st.error("❌ GOOGLE_API_KEY is missing. Add it in Streamlit Secrets.")
# #     st.stop()

# # configure(api_key=GOOGLE_API_KEY)
# # gemini = GenerativeModel("gemini-2.5-flash")

# # # ================================
# # # 📂 VECTOR DB PATHS
# # # ================================
# # BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# # PERSIST_DIR_1 = os.path.join(BASE_DIR, "chroma_db_nomic")
# # PERSIST_DIR_2 = os.path.join(BASE_DIR, "chroma_db_jsonl")

# # # ================================
# # # 🧠 LOAD VECTOR STORES
# # # ================================
# # embeddings = OllamaEmbeddings(model="nomic-embed-text")

# # db1 = Chroma(persist_directory=PERSIST_DIR_1, embedding_function=embeddings)
# # db2 = Chroma(persist_directory=PERSIST_DIR_2, embedding_function=embeddings)

# # retriever1 = db1.as_retriever(search_kwargs={"k": 8})
# # retriever2 = db2.as_retriever(search_kwargs={"k": 8})

# # # ================================
# # # 🌐 EXTERNAL TOOL: WIKIPEDIA
# # # ================================
# # wiki_tool = WikipediaQueryRun(api_wrapper=WikipediaAPIWrapper())

# # # ================================
# # # 💾 SIMPLE CHAT MEMORY
# # # ================================
# # MEMORY_FILE = "chat_memory.json"


# # def load_memory() -> List[Dict[str, str]]:
# #     if os.path.exists(MEMORY_FILE):
# #         try:
# #             with open(MEMORY_FILE, "r", encoding="utf-8") as f:
# #                 return json.load(f)
# #         except Exception:
# #             return []
# #     return []


# # def save_memory(mem: List[Dict[str, str]]) -> None:
# #     # keep last 15 turns
# #     with open(MEMORY_FILE, "w", encoding="utf-8") as f:
# #         json.dump(mem[-15:], f, indent=2)


# # # ================================
# # # 🔧 UTILITY FUNCTIONS
# # # ================================
# # def clean_query(q: str) -> str:
# #     return re.sub(r"[\n\r]+", " ", q.strip())


# # def gemini_call(prompt: str) -> str:
# #     """Safe wrapper around Gemini generate_content."""
# #     try:
# #         resp = gemini.generate_content(prompt)
# #         return (resp.text or "").strip()
# #     except Exception as e:
# #         return f"[LLM Error: {e}]"


# # def extractive_answer(query: str, docs: List[Any]) -> str:
# #     """Use RAG context to answer with inline numeric citations [1], [2], ..."""
# #     if not docs:
# #         return ""

# #     ctx = "\n\n".join(f"[{i+1}] {d.page_content}" for i, d in enumerate(docs[:6]))
# #     prompt = f"""
# # You are an expert assistant. Answer the question using ONLY the following CONTEXT.
# # Each sentence you write must end with a citation like [1], [2], etc. matching the context numbers.
# # If the context is insufficient, reply with NOINFO.

# # Question:
# # {query}

# # CONTEXT:
# # {ctx}
# # """
# #     ans = gemini_call(prompt)
# #     if ans.upper().startswith("NOINFO") or len(ans) < 40:
# #         return ""
# #     return ans


# # def scholarly_lookup(query: str, max_results: int = 3) -> List[str]:
# #     """Optional: fetch a few references from CrossRef for nicer citations."""
# #     refs: List[str] = []
# #     try:
# #         r = requests.get(
# #             f"https://api.crossref.org/works?rows={max_results}&query={quote(query)}",
# #             timeout=8,
# #         )
# #         data = r.json()
# #         for item in data.get("message", {}).get("items", []):
# #             title = item.get("title", ["Untitled"])[0]
# #             authors = item.get("author", [])
# #             author_str = ", ".join(a.get("family", "") for a in authors[:2]) or "Unknown"
# #             if len(authors) > 2:
# #                 author_str += " et al."
# #             year = item.get("issued", {}).get("date-parts", [[None]])[0][0]
# #             doi = item.get("DOI", "")
# #             link = f"https://doi.org/{doi}" if doi else item.get("URL", "")
# #             refs.append(f"{author_str} ({year}). *{title}*. {link}")
# #         if refs:
# #             return refs
# #     except Exception:
# #         pass
# #     return ["(No scholarly reference found)"]


# # def format_clickable_citations(citations: List[str]) -> str:
# #     out = []
# #     for i, c in enumerate(citations, 1):
# #         m = re.search(r"(https?://[^\s]+|doi\.org/[^\s)]+)", c)
# #         if m:
# #             out.append(f"[{i}] [{c}]({m.group(1)})")
# #         else:
# #             out.append(f"[{i}] {c}")
# #     return "\n".join(out)


# # # ================================
# # # 🔀 LANGGRAPH STATE & NODES
# # # ================================
# # class GraphState(TypedDict):
# #     query: str
# #     answer: str
# #     context: str  # "db1", "db2", "google", "wiki", "gbif", "inat", or ""
# #     citations: List[str]
# #     chat_history: List[Dict[str, str]]


# # def db1_node(state: GraphState) -> GraphState:
# #     q = clean_query(state["query"])
# #     docs = retriever1.invoke(q)  # ✅ new LC API: invoke()
# #     ans = extractive_answer(q, docs)
# #     return {**state, "context": "db1" if ans else "", "answer": ans}


# # def db2_node(state: GraphState) -> GraphState:
# #     q = clean_query(state["query"])
# #     docs = retriever2.invoke(q)  # ✅ new LC API: invoke()
# #     ans = extractive_answer(q, docs)
# #     return {**state, "context": "db2" if ans else "", "answer": ans}


# # def google_node(state: GraphState) -> GraphState:
# #     """
# #     'Google' step implemented via Gemini's web-augmented reasoning.
# #     No external GoogleSearchAPIWrapper required.
# #     """
# #     q = state["query"]
# #     prompt = f"""
# # You have access to web search tools.
# # Search the web (implicitly) for **recent and reliable** information and answer this question:

# # {q}

# # If you truly cannot find anything useful, reply with exactly: NOINFO
# # """
# #     ans = gemini_call(prompt)
# #     if ans.upper().startswith("NOINFO") or len(ans) < 40:
# #         return {**state, "context": "", "answer": state["answer"]}
# #     return {**state, "context": "google", "answer": ans}


# # def wiki_node(state: GraphState) -> GraphState:
# #     try:
# #         r = wiki_tool.run(state["query"])
# #         if r and len(r.strip()) > 40:
# #             return {**state, "context": "wiki", "answer": r}
# #     except Exception:
# #         pass
# #     return {**state, "context": "", "answer": state["answer"]}


# # def gbif_node(state: GraphState) -> GraphState:
# #     """
# #     Placeholder for future GBIF integration.
# #     Currently just passes state forward.
# #     """
# #     return state


# # def inat_node(state: GraphState) -> GraphState:
# #     """
# #     Placeholder for future iNaturalist integration.
# #     Currently just passes state forward.
# #     """
# #     return state


# # def final_node(state: GraphState) -> GraphState:
# #     """
# #     Final assembly: if we still don't have a good answer,
# #     let Gemini answer directly using its own knowledge + reasoning.
# #     Optionally add scholarly references.
# #     """
# #     if not state["answer"] or len(state["answer"]) < 40:
# #         direct_ans = gemini_call(
# #             f"Answer this question clearly, step-by-step:\n\n{state['query']}"
# #         )
# #         state["answer"] = direct_ans

# #     # Optional: attach a few references (not enforced)
# #     refs = scholarly_lookup(state["query"], max_results=2)
# #     state["citations"] = refs
# #     return state


# # # ================================
# # # 🧩 BUILD LANGGRAPH WORKFLOW
# # # ================================
# # workflow = StateGraph(GraphState)

# # workflow.add_node("db1", db1_node)
# # workflow.add_node("db2", db2_node)
# # workflow.add_node("google", google_node)
# # workflow.add_node("wiki", wiki_node)
# # workflow.add_node("gbif", gbif_node)
# # workflow.add_node("inat", inat_node)
# # workflow.add_node("final", final_node)

# # # START → DB1
# # workflow.add_edge(START, "db1")

# # # If DB1 answered → final, else → DB2
# # workflow.add_conditional_edges(
# #     "db1",
# #     lambda s: s["context"],
# #     {"db1": "final", "": "db2"},
# # )

# # # If DB2 answered → final, else → Google step (via Gemini)
# # workflow.add_conditional_edges(
# #     "db2",
# #     lambda s: s["context"],
# #     {"db2": "final", "": "google"},
# # )

# # # If Google step answered → final, else → Wikipedia
# # workflow.add_conditional_edges(
# #     "google",
# #     lambda s: s["context"],
# #     {"google": "final", "": "wiki"},
# # )

# # # If Wikipedia answered → final, else → GBIF → iNat → final
# # workflow.add_conditional_edges(
# #     "wiki",
# #     lambda s: s["context"],
# #     {"wiki": "final", "": "gbif"},
# # )

# # workflow.add_edge("gbif", "inat")
# # workflow.add_edge("inat", "final")

# # workflow.add_edge("final", END)

# # graph = workflow.compile()

# # # ================================
# # # 🎨 STREAMLIT UI
# # # ================================
# # st.set_page_config(
# #     page_title="Hybrid RAG Chatbot",
# #     page_icon="🤖",
# #     layout="wide",
# # )

# # st.title("🤖 Hybrid RAG Chatbot")
# # st.write(
# #     "This bot first searches your local knowledge bases (DB1 → DB2), "
# #     "then uses web search (via Gemini), Wikipedia, and finally Gemini's own reasoning."
# # )

# # if "chat" not in st.session_state:
# #     st.session_state.chat = load_memory()

# # user_input = st.text_input("Ask a question:")

# # if st.button("Ask"):
# #     if user_input.strip():
# #         mem = st.session_state.chat

# #         result = graph.invoke(
# #             {
# #                 "query": user_input,
# #                 "answer": "",
# #                 "context": "",
# #                 "citations": [],
# #                 "chat_history": mem,
# #             }
# #         )

# #         answer = result["answer"]
# #         context = result["context"]
# #         citations = result.get("citations", [])

# #         st.write("### 🧠 Answer")
# #         st.write(answer)

# #         if context:
# #             st.write(f"**Primary source step:** `{context}`")

# #         if citations:
# #             st.write("### 📚 References")
# #             st.markdown(format_clickable_citations(citations))

# #         mem.append({"query": user_input, "answer": answer})
# #         save_memory(mem)
# #         st.session_state.chat = mem

# # st.write("---")
# # st.write("### 💬 Recent Chat History")
# # for c in st.session_state.chat[-10:]:
# #     st.markdown(f"**You:** {c['query']}")
# #     st.markdown(f"**Bot:** {c['answer']}")





































# # import streamlit as st
# # import os
# # import json
# # import re
# # from urllib.parse import quote
# # from typing import TypedDict, List, Dict, Any
# # import requests

# # from langgraph.graph import StateGraph, START, END
# # from langchain_community.embeddings import OllamaEmbeddings
# # from langchain_community.vectorstores import Chroma
# # from langchain_community.utilities import WikipediaAPIWrapper, GoogleSearchAPIWrapper
# # from langchain_community.tools import WikipediaQueryRun, GoogleSearchRun

# # # ---------------------------
# # # 🔑 GOOGLE GENAI (DIRECT)
# # # ---------------------------
# # from google.generativeai import configure, GenerativeModel

# # if "GOOGLE_API_KEY" not in os.environ:
# #     st.error("❌ GOOGLE_API_KEY missing")
# # else:
# #     configure(api_key=os.environ["GOOGLE_API_KEY"])
# #     gemini = GenerativeModel("gemini-2.5-flash")

# # GOOGLE_CSE_ID = os.getenv("GOOGLE_CSE_ID")

# # # -------------------------------------------
# # # 📂 Vector DB Paths (relative)
# # # -------------------------------------------
# # BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# # PERSIST_DIR_1 = os.path.join(BASE_DIR, "chroma_db_nomic")
# # PERSIST_DIR_2 = os.path.join(BASE_DIR, "chroma_db_jsonl")

# # # -------------------------------------------
# # # 🧠 Load DBs
# # # -------------------------------------------
# # embeddings = OllamaEmbeddings(model="nomic-embed-text")

# # db1 = Chroma(persist_directory=PERSIST_DIR_1, embedding_function=embeddings)
# # db2 = Chroma(persist_directory=PERSIST_DIR_2, embedding_function=embeddings)

# # retriever1 = db1.as_retriever(search_kwargs={"k": 8})
# # retriever2 = db2.as_retriever(search_kwargs={"k": 8})

# # # -------------------------------------------
# # # 🌐 External Tools
# # # -------------------------------------------
# # wiki_tool = WikipediaQueryRun(api_wrapper=WikipediaAPIWrapper())
# # google_tool = GoogleSearchRun(api_wrapper=GoogleSearchAPIWrapper(
# #     google_api_key=os.environ["GOOGLE_API_KEY"], google_cse_id=GOOGLE_CSE_ID
# # ))

# # # -------------------------------------------
# # # 🧠 Memory Functions
# # # -------------------------------------------
# # MEMORY_FILE = "chat_memory.json"

# # def load_memory():
# #     if os.path.exists(MEMORY_FILE):
# #         try:
# #             return json.load(open(MEMORY_FILE, "r", encoding="utf-8"))
# #         except:
# #             return []
# #     return []

# # def save_memory(mem):
# #     json.dump(mem[-15:], open(MEMORY_FILE, "w", encoding="utf-8"), indent=2)

# # # -------------------------------------------
# # # Utility Functions
# # # -------------------------------------------
# # def clean_query(q: str) -> str:
# #     return re.sub(r"[\n\r]+", " ", q.strip())

# # def gemini_call(prompt: str) -> str:
# #     """Direct Gemini LLM call."""
# #     try:
# #         resp = gemini.generate_content(prompt)
# #         return resp.text.strip()
# #     except Exception as e:
# #         return f"[LLM Error: {e}]"

# # def extractive_answer(query: str, docs: List[Any]) -> str:
# #     ctx = "\n\n".join(f"[{i+1}] {d.page_content}" for i, d in enumerate(docs[:6]))
# #     prompt = f"""
# # Answer the question below **using only the provided CONTEXT**.
# # Each sentence must end with [1], [2], etc. referencing the context.
# # If context is insufficient, write NOINFO.

# # Question: {query}

# # CONTEXT:
# # {ctx}
# # """
# #     ans = gemini_call(prompt)
# #     if ans.upper().startswith("NOINFO") or len(ans) < 40:
# #         return ""
# #     return ans

# # def scholarly_lookup(query: str, max_results=3):
# #     refs = []
# #     try:
# #         r = requests.get(
# #             f"https://api.crossref.org/works?rows={max_results}&query={quote(query)}",
# #             timeout=8
# #         ).json()

# #         for item in r.get("message", {}).get("items", []):
# #             title = item.get("title", ["Untitled"])[0]
# #             authors = item.get("author", [])
# #             author_str = ", ".join(a.get("family", "") for a in authors[:2]) or "Unknown"
# #             if len(authors) > 2:
# #                 author_str += " et al."
# #             year = item.get("issued", {}).get("date-parts", [[None]])[0][0]
# #             doi = item.get("DOI", "")
# #             link = f"https://doi.org/{doi}" if doi else item.get("URL", "")
# #             refs.append(f"{author_str} ({year}). *{title}*. {link}")
# #         if refs:
# #             return refs
# #     except:
# #         pass
# #     return ["(No scholarly reference found)"]

# # def format_clickable_citations(citations: List[str]) -> str:
# #     out = []
# #     for i, c in enumerate(citations, 1):
# #         m = re.search(r"(https?://[^\s]+|doi\.org/[^\s)]+)", c)
# #         if m:
# #             out.append(f"[{i}] [{c}]({m.group(1)})")
# #         else:
# #             out.append(f"[{i}] {c}")
# #     return "\n".join(out)

# # # -------------------------------------------
# # # 🔀 Graph Workflow (nodes unchanged)
# # # -------------------------------------------
# # class GraphState(TypedDict):
# #     query: str
# #     answer: str
# #     context: str
# #     citations: List[str]
# #     chat_history: List[Dict[str, str]]

# # def db1_node(state: GraphState):
# #     q = clean_query(state["query"])
# #     docs = retriever1.get_relevant_documents(q)
# #     ans = extractive_answer(q, docs)
# #     return {**state, "context": "DB1" if ans else "", "answer": ans}

# # def db2_node(state: GraphState):
# #     q = clean_query(state["query"])
# #     docs = retriever2.get_relevant_documents(q)
# #     ans = extractive_answer(q, docs)
# #     return {**state, "context": "DB2" if ans else "", "answer": ans}

# # def google_node(state: GraphState):
# #     try:
# #         r = google_tool.run(state["query"])
# #         if r:
# #             return {**state, "context": "Google", "answer": r}
# #     except:
# #         pass
# #     return {**state, "context": ""}

# # def wiki_node(state: GraphState):
# #     try:
# #         r = wiki_tool.run(state["query"])
# #         if r:
# #             return {**state, "context": "Wikipedia", "answer": r}
# #     except:
# #         pass
# #     return {**state, "context": ""}

# # def gbif_node(state: GraphState):
# #     return state   # placeholder

# # def inat_node(state: GraphState):
# #     return state   # placeholder

# # def final_node(state: GraphState):
# #     if not state["answer"]:
# #         # fallback to generative answer
# #         state["answer"] = gemini_call(state["query"])
# #     return state

# # # Build workflow
# # workflow = StateGraph(GraphState)
# # workflow.add_node("db1", db1_node)
# # workflow.add_node("db2", db2_node)
# # workflow.add_node("google", google_node)
# # workflow.add_node("wiki", wiki_node)
# # workflow.add_node("gbif", gbif_node)
# # workflow.add_node("inat", inat_node)
# # workflow.add_node("final", final_node)

# # workflow.add_edge(START, "db1")
# # workflow.add_conditional_edges("db1", lambda s: s["context"], {"db1": "final", "": "db2"})
# # workflow.add_conditional_edges("db2", lambda s: s["context"], {"db2": "final", "": "google"})
# # workflow.add_conditional_edges("google", lambda s: s["context"], {"google": "final", "": "wiki"})
# # workflow.add_conditional_edges("wiki", lambda s: s["context"], {"wiki": "final", "": "gbif"})
# # workflow.add_edge("gbif", "inat")
# # workflow.add_edge("inat", "final")
# # graph = workflow.compile()

# # # -------------------------------------------
# # # 🎨 Streamlit UI
# # # -------------------------------------------
# # st.set_page_config(page_title="Hybrid RAG Chatbot", page_icon="🤖", layout="wide")
# # st.title("🤖 Hybrid RAG Chatbot")

# # if "chat" not in st.session_state:
# #     st.session_state.chat = load_memory()

# # user_input = st.text_input("Enter your question:")

# # if st.button("Ask"):
# #     if user_input.strip():
# #         mem = st.session_state.chat

# #         result = graph.invoke({
# #             "query": user_input,
# #             "answer": "",
# #             "context": "",
# #             "citations": [],
# #             "chat_history": mem
# #         })

# #         answer = result["answer"]
# #         context = result["context"]

# #         st.write("### 🧠 Answer")
# #         st.write(answer)
# #         st.write(f"**Source:** `{context}`")

# #         mem.append({"query": user_input, "answer": answer})
# #         save_memory(mem)
# #         st.session_state.chat = mem

# # st.write("---")
# # st.write("### 💬 Chat History")
# # for c in st.session_state.chat[-10:]:
# #     st.markdown(f"**You:** {c['query']}")
# #     st.markdown(f"**Bot:** {c['answer']}")































# # # import streamlit as st
# # # import os
# # # import json
# # # import re
# # # from urllib.parse import quote
# # # from typing import TypedDict, List, Dict, Any
# # # import requests

# # # from langgraph.graph import StateGraph, START, END
# # # from langchain_google_genai import ChatGoogleGenerativeAI
# # # from langchain_community.embeddings import OllamaEmbeddings
# # # from langchain_community.vectorstores import Chroma
# # # from langchain_community.utilities import WikipediaAPIWrapper, GoogleSearchAPIWrapper
# # # from langchain_community.tools import WikipediaQueryRun, GoogleSearchRun

# # # # -------------------------------------------
# # # # 🔑 API KEYS (Streamlit Secrets or OS Vars)
# # # # -------------------------------------------
# # # GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
# # # GOOGLE_CSE_ID = os.getenv("GOOGLE_CSE_ID")

# # # if not GOOGLE_API_KEY:
# # #     st.error("❌ GOOGLE_API_KEY is missing. Add it to your environment or Streamlit secrets.")
# # # if not GOOGLE_CSE_ID:
# # #     st.error("❌ GOOGLE_CSE_ID is missing.")

# # # # -------------------------------------------
# # # # 🤖 LLM
# # # # -------------------------------------------
# # # gemini = ChatGoogleGenerativeAI(
# # #     model="gemini-2.5-flash",
# # #     temperature=0,
# # #     api_key=GOOGLE_API_KEY
# # # )

# # # # -------------------------------------------
# # # # 📂 Vector DB Paths (relative)
# # # # -------------------------------------------
# # # BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# # # PERSIST_DIR_1 = os.path.join(BASE_DIR, "chroma_db_nomic")
# # # PERSIST_DIR_2 = os.path.join(BASE_DIR, "chroma_db_jsonl")

# # # # -------------------------------------------
# # # # 🧠 Load DBs
# # # # -------------------------------------------
# # # embeddings = OllamaEmbeddings(model="nomic-embed-text")

# # # db1 = Chroma(persist_directory=PERSIST_DIR_1, embedding_function=embeddings)
# # # db2 = Chroma(persist_directory=PERSIST_DIR_2, embedding_function=embeddings)

# # # retriever1 = db1.as_retriever(search_kwargs={"k": 8})
# # # retriever2 = db2.as_retriever(search_kwargs={"k": 8})

# # # # -------------------------------------------
# # # # 🌐 External Tools
# # # # -------------------------------------------
# # # wiki_tool = WikipediaQueryRun(api_wrapper=WikipediaAPIWrapper())
# # # google_tool = GoogleSearchRun(api_wrapper=GoogleSearchAPIWrapper(
# # #     google_api_key=GOOGLE_API_KEY, google_cse_id=Google_CSE_ID
# # # ))

# # # # -------------------------------------------
# # # # 🧠 Memory Functions
# # # # -------------------------------------------
# # # MEMORY_FILE = "chat_memory.json"

# # # def load_memory():
# # #     if os.path.exists(MEMORY_FILE):
# # #         try:
# # #             return json.load(open(MEMORY_FILE, "r", encoding="utf-8"))
# # #         except:
# # #             return []
# # #     return []

# # # def save_memory(mem):
# # #     json.dump(mem[-15:], open(MEMORY_FILE, "w", encoding="utf-8"), indent=2)

# # # # -------------------------------------------
# # # # Utility Functions (clean_query, extractive_answer, etc.)
# # # # -------------------------------------------
# # # def clean_query(q: str) -> str:
# # #     return re.sub(r"[\n\r]+", " ", q.strip())

# # # def extractive_answer(query: str, docs: List[Any]) -> str:
# # #     ctx = "\n\n".join(f"[{i+1}] {d.page_content}" for i, d in enumerate(docs[:6]))
# # #     prompt = f"""
# # # Answer the question below using only the provided CONTEXT.
# # # Each sentence should end with [1], [2], etc. referencing the numbered context.
# # # If context insufficient, write NOINFO.

# # # Question: {query}
# # # CONTEXT:
# # # {ctx}
# # # """
# # #     ans = gemini.invoke(prompt).content.strip()
# # #     if ans.upper().startswith("NOINFO") or len(ans) < 40:
# # #         return ""
# # #     return ans

# # # def scholarly_lookup(query: str, max_results=3):
# # #     refs = []
# # #     try:
# # #         r = requests.get(
# # #             f"https://api.crossref.org/works?rows={max_results}&query={quote(query)}",
# # #             timeout=8
# # #         ).json()
# # #         for item in r.get("message", {}).get("items", []):
# # #             title = item.get("title", ["Untitled"])[0]
# # #             authors = item.get("author", [])
# # #             author_str = ", ".join(a.get("family", "") for a in authors[:2]) or "Unknown"
# # #             if len(authors) > 2:
# # #                 author_str += " et al."
# # #             year = item.get("issued", {}).get("date-parts", [[None]])[0][0]
# # #             doi = item.get("DOI", "")
# # #             link = f"https://doi.org/{doi}" if doi else item.get("URL", "")
# # #             refs.append(f"{author_str} ({year}). *{title}*. {link}")
# # #         if refs:
# # #             return refs
# # #     except:
# # #         pass
# # #     return ["(No scholarly reference found)"]

# # # def format_clickable_citations(citations: List[str]) -> str:
# # #     out = []
# # #     for i, c in enumerate(citations, 1):
# # #         m = re.search(r"(https?://[^\s]+|doi\.org/[^\s)]+)", c)
# # #         if m:
# # #             out.append(f"[{i}] [{c}]({m.group(1)})")
# # #         else:
# # #             out.append(f"[{i}] {c}")
# # #     return "\n".join(out)

# # # # -------------------------------------------
# # # # 🔀 Graph Workflow
# # # # -------------------------------------------
# # # class GraphState(TypedDict):
# # #     query: str
# # #     answer: str
# # #     context: str
# # #     citations: List[str]
# # #     chat_history: List[Dict[str, str]]

# # # # Nodes are **100% the same** — reused from your script
# # # # (db1_node, db2_node, google_node, wiki_node, gbif_node, inat_node, final_node)

# # # # Build graph
# # # workflow = StateGraph(GraphState)
# # # workflow.add_node("db1", db1_node)
# # # workflow.add_node("db2", db2_node)
# # # workflow.add_node("google", google_node)
# # # workflow.add_node("wiki", wiki_node)
# # # workflow.add_node("gbif", gbif_node)
# # # workflow.add_node("inat", inat_node)
# # # workflow.add_node("final", final_node)

# # # workflow.add_edge(START, "db1")
# # # workflow.add_conditional_edges("db1", lambda s: s["context"], {"db1": "final", "no_db1": "db2"})
# # # workflow.add_conditional_edges("db2", lambda s: s["context"], {"db2": "final", "no_db2": "google"})
# # # workflow.add_conditional_edges("google", lambda s: s["context"], {"google": "final", "no_google": "wiki"})
# # # workflow.add_conditional_edges("wiki", lambda s: s["context"], {"wiki": "final", "no_wiki": "gbif"})
# # # workflow.add_conditional_edges("gbif", lambda s: s["context"], {"gbif": "final", "no_gbif": "inat"})
# # # workflow.add_edge("inat", "final")
# # # workflow.add_edge("final", END)

# # # graph = workflow.compile()

# # # # -------------------------------------------
# # # # 🎨 Streamlit UI
# # # # -------------------------------------------
# # # st.set_page_config(page_title="Hybrid RAG Chatbot", page_icon="🤖", layout="wide")

# # # st.title("🤖 Hybrid RAG Chatbot")
# # # st.write("Ask any question — the bot searches DB1 → DB2 → Google → Wikipedia → GBIF → iNaturalist.")

# # # if "chat" not in st.session_state:
# # #     st.session_state.chat = load_memory()

# # # user_input = st.text_input("Enter your question:")

# # # if st.button("Ask"):
# # #     if user_input.strip():
# # #         mem = st.session_state.chat

# # #         result = graph.invoke({
# # #             "query": user_input,
# # #             "answer": "",
# # #             "context": "",
# # #             "citations": [],
# # #             "chat_history": mem
# # #         })

# # #         answer = result["answer"]
# # #         context = result["context"]

# # #         st.write("### 🧠 Answer")
# # #         st.write(answer)
# # #         st.write(f"**Source:** `{context}`")

# # #         mem.append({"query": user_input, "answer": answer})
# # #         save_memory(mem)
# # #         st.session_state.chat = mem

# # # st.write("---")
# # # st.write("### 💬 Chat History")
# # # for c in st.session_state.chat[-10:]:
# # #     st.markdown(f"**You:** {c['query']}")
# # #     st.markdown(f"**Bot:** {c['answer']}")
