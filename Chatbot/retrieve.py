import os
import json
from typing import Any
from typing_extensions import TypedDict, Annotated

import faiss
import numpy as np
from dotenv import load_dotenv

# Google GenAI SDK
import google.generativeai as genai

from langchain_core.messages import HumanMessage, SystemMessage, AnyMessage
from langchain_google_genai import ChatGoogleGenerativeAI

from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.checkpoint.memory import MemorySaver
from config import *
# ===============================
# LOAD ENV
# ===============================
load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    raise RuntimeError("❌ GEMINI_API_KEY not found in .env file")

genai.configure(api_key=GEMINI_API_KEY)

# ===============================
# EMBEDDING CLIENT
# ===============================
from embedding import GeminiEmbeddingClient

embedder = GeminiEmbeddingClient(GEMINI_API_KEY)

# ===============================
# VECTOR SEARCH
# ===============================
class VectorSearch:
    def __init__(self, index_path: Path, metadata_path: Path):
        if not index_path.exists():
            raise FileNotFoundError("❌ FAISS index not found")
        if not metadata_path.exists():
            raise FileNotFoundError("❌ Metadata file not found")

        self.index = faiss.read_index(str(index_path))
        with open(metadata_path, "r", encoding="utf-8") as f:
            self.metadata = json.load(f)

        if self.index.d != EMBEDDING_DIM:
            raise ValueError(
                f"❌ FAISS dim ({self.index.d}) != EMBEDDING_DIM ({EMBEDDING_DIM})"
            )

        print(f"✅ FAISS cosine index loaded (dim={self.index.d})")

    def search(self, query_vector: np.ndarray):
        """Return top-k semantically similar chunks with scores"""
        if query_vector.ndim == 1:
            query_vector = query_vector.reshape(1, -1)

        faiss.normalize_L2(query_vector)
        scores, indices = self.index.search(query_vector, TOP_K)

        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx == -1 or score < SIMILARITY_THRESHOLD:
                continue
            results.append({
                "content": self.metadata[idx]["content"],
                "score": float(score)
            })
        return results

# ===============================
# GENERATIVE MODEL
# ===============================
class GeminiFlash:
    def __init__(self):
        self.model = genai.GenerativeModel(LLM_MODEL)

    def answer(self, question: str, context: list[dict]) -> str:
        """
        context: list of dicts {"content": str, "score": float}
        """
        if not context:
            return "I don't know"

        # Include context with similarity scores
        context_text = "\n\n".join(
            [f"[score: {c['score']:.3f}] {c['content']}" for c in context]
        )

        prompt = f"""
You are an AI assistant.

Use the information in the context below to answer the question.
If the context contains relevant information, explain or summarize it clearly.
If the context is not relevant to the question, say "I don't know".

Context:
{context_text}

Question:
{question}
"""
        response = self.model.generate_content(
            prompt,
            generation_config={
                "temperature": 0.0,
                "max_output_tokens": 2048
            }
        )

        if not response or not response.text:
            return "I don't know"

        return response.text.strip()

# ===============================
# QA SYSTEM
# ===============================
class QASystem:
    def __init__(self):
        self.embedder = GeminiEmbeddingClient(GEMINI_API_KEY)
        self.vector_search = VectorSearch(
            FAISS_INDEX_PATH,
            FAISS_META_PATH
        )
        self.generator = GeminiFlash()

    def ask(self, question: str) -> str:
        print("🔎 Embedding query...")
        query_vec = np.array(self.embedder.embed(question), dtype="float32")

        if query_vec.size == 0:
            raise ValueError("❌ Empty embedding returned")

        print("📂 Semantic similarity search...")
        results = self.vector_search.search(query_vec)

        if not results:
            print("⚠️ No semantically similar content found")
            return "I don't know"

        print("\n🔎 Top semantic matches:")
        for r in results:
            print(f"  • cosine_score = {r['score']:.3f} | {r['content'][:80]}...")

        print("🤖 Generating answer...")
        return self.generator.answer(question, results)

# ===============================
# LANGGRAPH RAG TOOL
# ===============================
class MessagesState(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]

def rag_tool(question: str) -> str:
    """
    LangGraph tool: retrieves FAISS context using embeddings and passes it to Gemini.
    """
    qa = QASystem()
    return qa.ask(question)

tools = [rag_tool]

# ===============================
# GEMINI CHAT MODEL (LangGraph)
# ===============================
llm = ChatGoogleGenerativeAI(
    model="models/gemini-2.5-flash-lite",
    temperature=1.0,
    max_tokens=None,
    timeout=None,
    max_retries=2,
)

sys_msg = SystemMessage(
    content=(
        "You are a helpful AI assistant.\n"
        "Use the provided context to answer the user's question.\n"
        "If the context is not relevant, say: 'I don't know'."
    )
)

def assistant(state: MessagesState):
    return {"messages": [llm.invoke([sys_msg] + state["messages"])]}

# ===============================
# BUILD STATEGRAPH
# ===============================
builder = StateGraph(MessagesState)
builder.add_node("assistant", assistant)
builder.add_node("tools", ToolNode(tools))
builder.add_edge(START, "assistant")
builder.add_conditional_edges("assistant", tools_condition)
builder.add_edge("tools", "assistant")

memory = MemorySaver()
react_graph = builder.compile(checkpointer=memory)

# ===============================
# RUN LOOP
# ===============================
if __name__ == "__main__":
    config = {"configurable": {"thread_id": "rag-session-1"}}
    print("✅ Gemini LangGraph RAG ready")

    while True:
        q = input("\n📝 Question (or exit): ").strip()
        if q.lower() in {"exit", "quit"}:
            break

        result = react_graph.invoke(
            {"messages": [HumanMessage(content=q)]},
            config
        )

        print("\n🤖 Answer:\n")
        for msg in result["messages"]:
            msg.pretty_print()
