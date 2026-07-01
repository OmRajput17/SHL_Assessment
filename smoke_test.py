"""Quick smoke test — run from shl_agent/ directory."""
import sys, os
# Load .env if present
try:
    from dotenv import load_dotenv
    load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), '..', '.env'))
except ImportError:
    pass

from app.catalog import load_catalog, find_by_name
from app.retrieval import search
from app.schemas import ChatRequest, Message

catalog = load_catalog()
assert len(catalog) >= 70, f"Expected >=70 records, got {len(catalog)}"
print(f"[OK] Catalog: {len(catalog)} records")

results = search("Java developer stakeholders", top_k=5)
names = [r["name"] for r in results]
assert any("Java" in n for n in names), f"Java not in top-5: {names}"
print(f"[OK] Retrieval (java developer): {names}")

results = search("personality questionnaire manager", top_k=5)
types = [r.get("test_type") for r in results]
assert "P" in types, f"No personality type in top-5: {types}"
print(f"[OK] Retrieval (personality manager): {[r['name'] for r in results]}")

rec = find_by_name("Java 8 New")
assert rec is not None, "Fuzzy resolve failed for 'Java 8 New'"
print(f"[OK] Fuzzy resolve 'Java 8 New' -> {rec['name']}")

rec = find_by_name("OPQ")
assert rec is not None, "Fuzzy resolve failed for 'OPQ'"
print(f"[OK] Fuzzy resolve 'OPQ' -> {rec['name']}")

rec = find_by_name("NonExistentXYZQQQ")
assert rec is None, "Expected None for unknown name"
print("[OK] Unknown name correctly returns None")

req = ChatRequest(messages=[Message(role="user", content="I need a Java dev")])
assert req.messages[0].content == "I need a Java dev"
print("[OK] Schema validation works")

print("\nAll smoke tests passed.")
