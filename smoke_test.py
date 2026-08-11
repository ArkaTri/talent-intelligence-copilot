# Verifikasi import dari contoh main.py masih valid di versi terpasang
try:
    from langchain_openai import ChatOpenAI, OpenAIEmbeddings
    print("OK  langchain_openai")
except Exception as e:
    print(f"GAGAL  langchain_openai: {e}")

try:
    from langchain_qdrant import QdrantVectorStore
    print("OK  langchain_qdrant")
except Exception as e:
    print(f"GAGAL  langchain_qdrant: {e}")

try:
    from langchain.tools import tool
    print("OK  langchain.tools.tool")
except Exception as e:
    print(f"GAGAL  langchain.tools.tool: {e}")

try:
    from langgraph.prebuilt import create_react_agent
    print("OK  langgraph.prebuilt.create_react_agent")
except Exception as e:
    print(f"GAGAL  create_react_agent: {e}")

try:
    from langchain_core.messages import ToolMessage
    print("OK  langchain_core.messages")
except Exception as e:
    print(f"GAGAL  langchain_core.messages: {e}")