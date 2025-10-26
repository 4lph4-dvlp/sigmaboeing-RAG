# doc2vdb.py
"""
CLI 사용법:
    python doc2vdb.py myfile.txt
    python doc2vdb.py document.md
"""

import sys
import os
from sentence_transformers import SentenceTransformer
import faiss
import pickle

def make_vector_db(file_path):
    if not os.path.exists(file_path):
        print(f"[!] 파일을 찾을 수 없습니다: {file_path}")
        return

    # === 파일 이름 및 출력 디렉토리 설정 ===
    base_name = os.path.splitext(os.path.basename(file_path))[0]
    out_dir = os.path.join("vdbs", base_name)
    os.makedirs(out_dir, exist_ok=True)

    vector_file = os.path.join(out_dir, "vector.index")
    meta_file = os.path.join(out_dir, "metadata.pkl")

    print(f"[*] 파일 처리 중: {file_path}")
    print(f"[*] 출력 디렉토리: {out_dir}")

    # === 파일 내용 읽기 ===
    with open(file_path, "r", encoding="utf-8") as f:
        text = f.read()

    # === 간단한 문단 분할 ===
    chunks = [t.strip() for t in text.split("\n") if t.strip()]

    # === SentenceTransformer로 임베딩 생성 ===
    model = SentenceTransformer("all-MiniLM-L6-v2")
    embeddings = model.encode(chunks)

    # === FAISS 인덱스 생성 ===
    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(embeddings)

    # === 저장 ===
    faiss.write_index(index, vector_file)
    with open(meta_file, "wb") as f:
        pickle.dump(chunks, f)

    print(f"[*] 벡터 DB 생성 완료: {vector_file}")
    print(f"[*] 총 문단 수: {len(chunks)}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("[!] 사용법: python doc2vdb.py <파일명.txt | 파일명.md>")
        sys.exit(1)

    file_path = sys.argv[1]
    make_vector_db(file_path)

