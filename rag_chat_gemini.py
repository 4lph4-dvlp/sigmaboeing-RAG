# rag_chat_gemini.py
"""
vdbs 내 모든 문서를 참고하여 Gemini 모델을 사용하는 RAG 챗봇
"""

import os
import faiss
import pickle
import numpy as np
from sentence_transformers import SentenceTransformer
import google.generativeai as genai

# === 설정 ===
VDB_DIR = "vdbs"

# === Gemini API 설정 ===
if "GOOGLE_API_KEY" not in os.environ:
    raise ValueError("[!] GOOGLE_API_KEY 환경변수를 설정해주세요.")
genai.configure(api_key=os.environ["GOOGLE_API_KEY"])

# === 사용할 모델 선택 ===
available_models = [
    "gemini-2.5-flash",
    "gemini-2.5-pro",
]
print("=== Gemini 모델 선택 ===")
for i, m in enumerate(available_models, 1):
    print(f"{i}. {m}")
choice = input("번호 선택: ").strip()
model_name = available_models[int(choice) - 1]
model = genai.GenerativeModel(model_name)
print(f"[*] 선택된 모델: {model_name}")

# === FAISS + 문서 로딩 ===
embedder = SentenceTransformer("all-MiniLM-L6-v2")

indexes = []
chunks_all = []

print("[*] 벡터 DB 로딩 중...")
for folder in os.listdir(VDB_DIR):
    subdir = os.path.join(VDB_DIR, folder)
    if not os.path.isdir(subdir):
        continue

    vpath = os.path.join(subdir, "vector.index")
    mpath = os.path.join(subdir, "metadata.pkl")

    if not (os.path.exists(vpath) and os.path.exists(mpath)):
        continue

    idx = faiss.read_index(vpath)
    with open(mpath, "rb") as f:
        chunks = pickle.load(f)

    indexes.append(idx)
    chunks_all.append(chunks)

if not indexes:
    print("[!] vdbs 폴더에 변환된 문서가 없습니다.")
    exit()

merged_index = faiss.IndexFlatL2(indexes[0].d)
merged_chunks = []
for idx, ch in zip(indexes, chunks_all):
    vecs = idx.reconstruct_n(0, idx.ntotal)
    merged_index.add(vecs)
    merged_chunks.extend(ch)

print(f"[*] 총 문단 수: {len(merged_chunks)}")

print("[*] RAG 챗봇 시작. 종료하려면 'exit' 입력.")
# === RAG 루프 ===
while True:
    query = input("\n[User] >> ").strip()
    if query.lower() in ["exit", "quit", "종료"]:
        break

    q_emb = embedder.encode([query])
    D, I = merged_index.search(q_emb, k=3)
    context = "\n".join([merged_chunks[i] for i in I[0]])

    prompt = f"""
다음은 검색된 관련 문서 내용입니다:
{context}

위 내용을 참고하여 아래 질문에 대해 한국어로 명확히 답변해주세요:
{query}
"""

    response = model.generate_content(prompt)
    print("\n[AI] : ", response.text.strip())

