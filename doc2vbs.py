# doc2vbs.py

import sys
import os
from sentence_transformers import SentenceTransformer
import faiss
import pickle
# ★ LangChain 스플리터 추가 ★
try:
    from langchain_text_splitters import RecursiveCharacterTextSplitter
except ImportError:
    print("[!] LangChain 라이브러리가 설치되지 않았습니다.")
    print("    터미널에 'py -m pip install langchain-text-splitters' 를 입력하여 설치해주세요.")
    sys.exit(1)

def make_vector_db(file_path):
    if not os.path.exists(file_path):
        print(f"[!] 파일을 찾을 수 없습니다: {file_path}")
        return

    # === 파일 이름 및 출력 디렉토리 설정 ===
    base_name = os.path.splitext(os.path.basename(file_path))[0]
    out_dir = os.path.join("vdbs", base_name)
    os.makedirs(out_dir, exist_ok=True) # 폴더가 없으면 만들기

    vector_file = os.path.join(out_dir, "vector.index")
    meta_file = os.path.join(out_dir, "metadata.pkl")

    print(f"[*] 파일 처리 중: {file_path}")
    print(f"[*] 출력 디렉토리: {out_dir}")

    # === 파일 내용 읽기 ===
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            text = f.read()
        print("[*] 파일 읽기 완료.")
    except Exception as e:
        print(f"[!] 파일 읽기 중 오류 발생: {e}")
        return

    # === ★ RecursiveCharacterTextSplitter로 텍스트 쪼개기 ★ ===
    print("[*] 텍스트 쪼개는 중 (RecursiveCharacterTextSplitter)...")
    try:
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1500,     # 조각당 최대 글자 수 (조절 가능)
            chunk_overlap=300,   # 조각끼리 겹치는 글자 수 (조절 가능)
            length_function=len,
            is_separator_regex=False,
            # 문단 -> 줄바꿈 -> 마침표/공백 -> 공백 -> 글자 순으로 나누기 시도
            separators=["\n\n", "\n", ". ", " ", ""],
        )
        # LangChain Document 객체 리스트 생성
        docs = text_splitter.create_documents([text])
        # Document 객체에서 텍스트 내용만 추출하여 리스트로 만듦
        chunks = [doc.page_content for doc in docs if doc.page_content.strip()] # 내용이 있는 조각만 포함

        if not chunks:
            print("[!] 텍스트를 조각으로 나눌 수 없습니다. 파일 내용을 확인해주세요.")
            return

        print(f"[*] 텍스트 쪼개기 완료. 총 조각 수: {len(chunks)}")

    except Exception as e:
        print(f"[!] 텍스트 쪼개기 중 오류 발생: {e}")
        return

    # === SentenceTransformer로 임베딩 생성 ===
    print("[*] 임베딩 생성 중 (SentenceTransformer)...")
    try:
        # 모델 로드 (인터넷 연결 필요 시 자동으로 다운로드)
        model = SentenceTransformer("all-MiniLM-L6-v2")
        # 임베딩 생성 (show_progress_bar=True 로 진행률 표시)
        embeddings = model.encode(chunks, show_progress_bar=True)
        print("[*] 임베딩 생성 완료.")
    except Exception as e:
        print(f"[!] 임베딩 생성 중 오류 발생: {e}")
        return

    # === FAISS 인덱스 생성 ===
    print("[*] FAISS 인덱스 생성 중...")
    try:
        dim = embeddings.shape[1] # 임베딩 벡터의 차원 확인
        index = faiss.IndexFlatL2(dim) # L2 거리 기반 인덱스 생성
        index.add(embeddings) # 생성된 임베딩을 인덱스에 추가
        print("[*] FAISS 인덱스 생성 완료.")
    except Exception as e:
        print(f"[!] FAISS 인덱스 생성 중 오류 발생: {e}")
        return

    # === 저장 ===
    print("[*] 벡터 DB 저장 중...")
    try:
        # FAISS 인덱스 파일 저장
        faiss.write_index(index, vector_file)
        # 원본 텍스트 조각(chunks) 리스트를 pickle 파일로 저장
        with open(meta_file, "wb") as f:
            pickle.dump(chunks, f)
        print(f"[*] 벡터 DB 저장 완료: {vector_file}")
        print(f"[*] 총 조각 수: {len(chunks)}")
    except Exception as e:
        print(f"[!] 벡터 DB 저장 중 오류 발생: {e}")
        return


if __name__ == "__main__":
    # ★★★ 여기에 처리할 파일의 전체 경로를 직접 입력 ★★★
    # (주의: r"..." 형태로 작성하고, \ 기호를 그대로 사용하세요)
    file_path = r"C:\Users\cheah\OneDrive\문서\대학 자료\2025보잉\pt2\Converted-Safety-Guide.md"

    # LangChain 라이브러리 설치 확인 (개선 코드에 있던 부분)
    # (이미 try-except 블록이 맨 위에 있어서 여기서는 생략해도 됩니다.)

    print(f"[*] 지정된 파일 처리 시작: {file_path}")
    # 벡터 DB 생성 함수 호출
    make_vector_db(file_path)