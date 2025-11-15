import os
os.environ["google_api_key"] = "AIzaSyBMSTBqYv74VqltxMj7G8eUtbuQg8tUROg"
os.environ["google_cse_id"] = "94a6404e7eb494900"



# =====================================================
# 🌟 FINAL HYBRID CHATBOT: Full Pipeline + Clickable Citations
# =====================================================

import os, re, json, requests
from urllib.parse import quote
from typing import TypedDict, List, Dict, Any
from langgraph.graph import StateGraph, START, END
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.utilities import WikipediaAPIWrapper, GoogleSearchAPIWrapper
from langchain_community.tools import WikipediaQueryRun, GoogleSearchRun

# -----------------------------
# 🔑 API KEYS
# -----------------------------
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
GOOGLE_CSE_ID = os.getenv("GOOGLE_CSE_ID")

# -----------------------------
# 🤖 LLM
# -----------------------------
gemini = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0, api_key=GOOGLE_API_KEY)

# -----------------------------
# 🧠 Vector DBs
# -----------------------------
PERSIST_DIR_1 = r"C:\Users\sadika957\Desktop\chatbot\scripts\chroma_db_nomic"
PERSIST_DIR_2 = r"C:\Users\sadika957\Desktop\chatbot\scripts\chroma_db_jsonl"

embeddings = OllamaEmbeddings(model="nomic-embed-text")

db1 = Chroma(persist_directory=PERSIST_DIR_1, embedding_function=embeddings)
db2 = Chroma(persist_directory=PERSIST_DIR_2, embedding_function=embeddings)
retriever1 = db1.as_retriever(search_kwargs={"k": 8})
retriever2 = db2.as_retriever(search_kwargs={"k": 8})

# -----------------------------
# 🌐 External Tools
# -----------------------------
wiki_tool = WikipediaQueryRun(api_wrapper=WikipediaAPIWrapper())
google_tool = GoogleSearchRun(api_wrapper=GoogleSearchAPIWrapper(
    google_api_key=GOOGLE_API_KEY, google_cse_id=GOOGLE_CSE_ID
))

# -----------------------------
# 🗂️ Memory
# -----------------------------
MEMORY_FILE = "chat_memory.json"

def load_memory() -> List[Dict[str, str]]:
    if os.path.exists(MEMORY_FILE):
        try:
            return json.load(open(MEMORY_FILE, "r", encoding="utf-8"))
        except:
            return []
    return []

def save_memory(mem: List[Dict[str, str]]):
    json.dump(mem[-15:], open(MEMORY_FILE, "w", encoding="utf-8"), indent=2)

# -----------------------------
# 🧩 Utility Functions
# -----------------------------
def clean_query(q: str) -> str:
    return re.sub(r"[\n\r]+", " ", q.strip())

def extractive_answer(query: str, docs: List[Any]) -> str:
    """LLM extracts directly from context"""
    ctx = "\n\n".join(f"[{i+1}] {d.page_content}" for i, d in enumerate(docs[:6]))
    prompt = f"""
Answer the question below using only the provided CONTEXT.
Each sentence should end with [1], [2], etc. referencing the numbered context.
If context insufficient, write NOINFO.

Question: {query}
CONTEXT:
{ctx}
"""
    ans = gemini.invoke(prompt).content.strip()
    if ans.upper().startswith("NOINFO") or len(ans) < 40:
        return ""
    return ans

def scholarly_lookup(query: str, max_results=3):
    """Fetch scholarly refs (CrossRef → Semantic Scholar fallback)"""
    citations = []
    try:
        r = requests.get(f"https://api.crossref.org/works?rows={max_results}&query={quote(query)}", timeout=8).json()
        for item in r.get("message", {}).get("items", []):
            title = item.get("title", ["Untitled"])[0]
            authors = item.get("author", [])
            author_str = ", ".join(a.get("family", "") for a in authors[:2]) or "Unknown"
            if len(authors) > 2:
                author_str += " et al."
            year = item.get("issued", {}).get("date-parts", [[None]])[0][0]
            doi = item.get("DOI", "")
            link = f"https://doi.org/{doi}" if doi else item.get("URL", "")
            citations.append(f"{author_str} ({year}). *{title}*. {link}")
        if citations:
            return citations
    except:
        pass
    try:
        s2 = requests.get(
            f"https://api.semanticscholar.org/graph/v1/paper/search?query={quote(query)}&limit={max_results}&fields=title,authors,year,url",
            timeout=8).json()
        for item in s2.get("data", []):
            title = item.get("title", "Untitled")
            authors = item.get("authors", [])
            author_str = ", ".join(a.get("name", "") for a in authors[:2]) or "Unknown"
            if len(authors) > 2:
                author_str += " et al."
            year = item.get("year", "n.d.")
            url = item.get("url", "")
            citations.append(f"{author_str} ({year}). *{title}*. {url}")
    except:
        pass
    return citations or ["(No scholarly reference found)"]

# 🖇️ Make citations clickable
def format_clickable_citations(citations: List[str]) -> str:
    """Converts citations to clickable markdown links"""
    formatted = []
    for i, c in enumerate(citations, start=1):
        m = re.search(r'(https?://[^\s]+|doi\.org/[^\s)]+)', c)
        if m:
            link = m.group(1).rstrip('.,)')
            title_match = re.search(r"\*([^*]+)\*", c)
            title = title_match.group(1) if title_match else f"Source {i}"
            formatted.append(f"[{i}] [{title}]({link})")
        else:
            google_link = f"https://www.google.com/search?q={quote(c)}"
            formatted.append(f"[{i}] [Search on Google]({google_link})")
    return "\n".join(formatted)

# -----------------------------
# 📚 Graph State
# -----------------------------
class GraphState(TypedDict):
    query: str
    answer: str
    context: str
    citations: List[str]
    chat_history: List[Dict[str, str]]

# -----------------------------
# 🧱 Nodes
# -----------------------------
def db1_node(state: GraphState):
    print("🔎 DB1...")
    q = clean_query(state["query"])
    try:
        docs = retriever1.get_relevant_documents(q)
    except:
        docs = []
    if not docs:
        return {**state, "context": "no_db1"}
    ans = extractive_answer(q, docs)
    if not ans:
        return {**state, "context": "no_db1"}
    link = f"[Search on Google](https://www.google.com/search?q={quote(q)})"
    ans += f"\n\n📚 Citations:\n{link}"
    return {**state, "answer": ans, "context": "db1", "citations": [link]}

def db2_node(state: GraphState):
    print("🔎 DB2 (research)...")
    q = clean_query(state["query"])
    try:
        docs = retriever2.get_relevant_documents(q)
    except:
        docs = []
    if not docs:
        return {**state, "context": "no_db2"}
    ans = extractive_answer(q, docs)
    if not ans:
        return {**state, "context": "no_db2"}
    refs = scholarly_lookup(q)
    clickable_refs = format_clickable_citations(refs[:3])
    ans += f"\n\n📚 Citations:\n{clickable_refs}"
    return {**state, "answer": ans, "context": "db2", "citations": refs}

def google_node(state: GraphState):
    print("🌐 Google...")
    q = clean_query(state["query"])
    raw = google_tool.run(q)
    if not raw:
        return {**state, "context": "no_google"}
    link = f"[Search on Google](https://www.google.com/search?q={quote(q)})"
    ans = gemini.invoke(f"Answer this based on Google snippets:\n{raw}").content.strip()
    ans += f"\n\n📚 Citations:\n{link}"
    return {**state, "answer": ans, "context": "google", "citations": [link]}

def wiki_node(state: GraphState):
    print("📖 Wikipedia...")
    q = clean_query(state["query"])
    blob = wiki_tool.run(q)
    if not blob:
        return {**state, "context": "no_wiki"}
    link = f"[Wikipedia Search](https://en.wikipedia.org/wiki/Special:Search?search={quote(q)})"
    ans = gemini.invoke(f"Answer using Wikipedia content:\n{blob}").content.strip()
    ans += f"\n\n📚 Citations:\n{link}"
    return {**state, "answer": ans, "context": "wiki", "citations": [link]}

def gbif_node(state: GraphState):
    print("🌍 GBIF...")
    q = clean_query(state["query"])
    url = f"https://api.gbif.org/v1/species/search?q={quote(q)}"
    try:
        r = requests.get(url, timeout=8).json()
        results = r.get("results", [])
        if not results:
            raise ValueError
        lines = [f"{it.get('scientificName','Unknown')} – https://www.gbif.org/species/{it.get('key','')}" for it in results[:5]]
        link = f"[GBIF Search](https://www.gbif.org/species/search?q={quote(q)})"
        ans = "\n".join(lines) + f"\n\n📚 Citations:\n{link}"
        return {**state, "answer": ans, "context": "gbif", "citations": [link]}
    except:
        return {**state, "context": "no_gbif"}

def inat_node(state: GraphState):
    print("🐝 iNaturalist...")
    q = clean_query(state["query"])
    url = f"https://api.inaturalist.org/v1/taxa/autocomplete?q={quote(q)}"
    try:
        r = requests.get(url, timeout=8).json()
        results = r.get("results", [])
        if not results:
            raise ValueError
        lines = [f"{it.get('name')} – https://www.inaturalist.org/taxa/{it.get('id')}" for it in results[:5]]
        link = f"[iNaturalist Search](https://www.inaturalist.org/search?q={quote(q)})"
        ans = "\n".join(lines) + f"\n\n📚 Citations:\n{link}"
        return {**state, "answer": ans, "context": "inat", "citations": [link]}
    except:
        return {**state, "context": "no_inat"}

# -----------------------------
# 🧠 Final Summarization
# -----------------------------
def final_node(state: GraphState):
    print("🧠 Summarizing...")
    q = clean_query(state["query"])
    base_answer = state["answer"].strip()
    citations = state.get("citations", [])
    summary_prompt = f"""
Summarize the following into a concise, well-structured, factual answer.
Preserve key technical details and remain accurate.

Question: {q}

Answer:
{base_answer}
"""
    summary = gemini.invoke(summary_prompt).content.strip()
    if citations:
        formatted_cits = format_clickable_citations(citations)
        summary += f"\n\n📚 Citations:\n{formatted_cits}"
    return {**state, "answer": summary, "context": state["context"]}

# -----------------------------
# 🔀 Graph Construction
# -----------------------------
workflow = StateGraph(GraphState)
workflow.add_node("db1", db1_node)
workflow.add_node("db2", db2_node)
workflow.add_node("google", google_node)
workflow.add_node("wiki", wiki_node)
workflow.add_node("gbif", gbif_node)
workflow.add_node("inat", inat_node)
workflow.add_node("final", final_node)

workflow.add_edge(START, "db1")
workflow.add_conditional_edges("db1", lambda s: s["context"], {"db1": "final", "no_db1": "db2"})
workflow.add_conditional_edges("db2", lambda s: s["context"], {"db2": "final", "no_db2": "google"})
workflow.add_conditional_edges("google", lambda s: s["context"], {"google": "final", "no_google": "wiki"})
workflow.add_conditional_edges("wiki", lambda s: s["context"], {"wiki": "final", "no_wiki": "gbif"})
workflow.add_conditional_edges("gbif", lambda s: s["context"], {"gbif": "final", "no_gbif": "inat"})
workflow.add_edge("inat", "final")
workflow.add_edge("final", END)
graph = workflow.compile()

print("✅ Pipeline ready: DB1 → DB2 → Google → Wiki → GBIF → iNat → Summarize")

def resolve_context_pronouns(query: str, memory: List[Dict[str, str]]) -> str:
    """Expands pronouns like 'it', 'they', etc. based on recent conversation memory."""
    if not memory:
        return query

    # Find most recent meaningful topic
    recent_answers = [m.get("answer", "") for m in memory[-3:] if m.get("answer")]
    if not recent_answers:
        return query

    context_text = " ".join(recent_answers[-3:])
    context_prompt = f"""
You are a contextual assistant. Replace vague pronouns (like it, they, them, he, she, etc.)
in the user's question with the specific subject from the recent conversation.

Example:
- Context: "BeeMachine is a platform that identifies bees."
- Query: "Who developed it?"
- Output: "Who developed BeeMachine?"

Now do this for:
Context: {context_text}
Query: {query}

Output only the rewritten question:
"""
    rewritten = gemini.invoke(context_prompt).content.strip()
    if len(rewritten) < 5 or rewritten.lower() == query.lower():
        return query
    return rewritten

# -----------------------------
def ask(question: str):
    mem = load_memory()

    # 🔍 Step 1: Contextual rewrite
    resolved_query = resolve_context_pronouns(question, mem)
    if resolved_query != question:
        print(f"💡 Interpreted query as: “{resolved_query}”")

    # 🔄 Step 2: Run pipeline
    result = graph.invoke({
        "query": resolved_query,
        "answer": "",
        "context": "",
        "citations": [],
        "chat_history": mem
    })

    # 💬 Step 3: Output
    print("\nChatbot:\n")
    print(result["answer"])
    print(f"\n🧭 Pipeline completed at source: {result['context']}")

    # 🧠 Step 4: Update memory
    mem.append({"query": question, "resolved_query": resolved_query, "answer": result["answer"]})
    save_memory(mem)
