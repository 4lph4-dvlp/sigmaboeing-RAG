# rag_chat_gemini.py
"""
vdbs 내 모든 문서를 참고하여 Gemini 모델을 사용하는 RAG 챗봇 (응답 시간, 참고 문헌 표시)
"""

import os
import faiss
import pickle
import numpy as np
from sentence_transformers import SentenceTransformer
import google.generativeai as genai
import time # ★ 추가 ★ 시간 측정을 위한 라이브러리

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

# 입력 유효성 검사 추가
while True:
    try:
        choice = input("번호 선택: ").strip()
        model_idx = int(choice) - 1
        if 0 <= model_idx < len(available_models):
            model_name = available_models[model_idx]
            break
        else:
            print("[!] 잘못된 번호입니다. 다시 입력해주세요.")
    except ValueError:
        print("[!] 숫자를 입력해주세요.")

try:
    # 모델 가용성 확인 로직 추가
    print(f"[*] 선택된 모델 '{model_name}' 확인 중...")
    model_info = genai.get_model(f'models/{model_name}') # 모델 정보 가져오기 시도
    if 'generateContent' not in model_info.supported_generation_methods:
        print(f"[!] 경고: 선택된 모델 '{model_name}'은 'generateContent'를 지원하지 않을 수 있습니다.")

    model = genai.GenerativeModel(model_name)
    print(f"[*] 선택된 모델: {model_name}")
except Exception as e:
    print(f"[!] 모델 ('{model_name}') 초기화 중 오류 발생: {e}")
    print("   - API 키가 유효한지, 선택한 모델 이름이 정확한지, 해당 모델 접근 권한이 있는지 확인해보세요.")
    print("   - 라이브러리를 최신 버전으로 업데이트했는지 확인하세요 (`py -m pip install --upgrade google-generativeai`)")
    exit()

# === FAISS + 문서 로딩 ===
print("[*] 임베딩 모델 로딩 중 (첫 실행 시 시간이 걸릴 수 있습니다)...")
try:
    embedder = SentenceTransformer("all-MiniLM-L6-v2")
    print("[*] 임베딩 모델 로딩 완료.")
except Exception as e:
     print(f"[!] 임베딩 모델 로딩 중 오류 발생: {e}")
     print("   - 인터넷 연결을 확인하거나 sentence-transformers 라이브러리를 재설치해보세요.")
     exit()


indexes = []
chunks_all = []
doc_sources = [] # 문서 출처(파일명) 저장용

print("[*] 벡터 DB 로딩 중...")
for folder in os.listdir(VDB_DIR):
    subdir = os.path.join(VDB_DIR, folder)
    if not os.path.isdir(subdir):
        continue

    vpath = os.path.join(subdir, "vector.index")
    mpath = os.path.join(subdir, "metadata.pkl")

    if not (os.path.exists(vpath) and os.path.exists(mpath)):
        print(f"   - [경고] '{subdir}' 폴더에 vector.index 또는 metadata.pkl 파일이 없습니다. 건너<0xEB><0><0x8E><0xB0>니다.")
        continue

    try:
        idx = faiss.read_index(vpath)
        with open(mpath, "rb") as f:
            chunks = pickle.load(f)

        if idx.ntotal != len(chunks):
             print(f"   - [경고] '{folder}'의 벡터 수({idx.ntotal})와 텍스트 조각 수({len(chunks)})가 일치하지 않습니다. 건너<0xEB><0><0x8E><0xB0>니다.")
             continue

        indexes.append(idx)
        chunks_all.append(chunks)
        doc_sources.extend([folder] * len(chunks))
        print(f"   - '{folder}' 로딩 완료 ({len(chunks)}개 조각)")

    except Exception as e:
        print(f"   - [오류] '{folder}' 로딩 중 오류 발생: {e}")


if not indexes:
    print("[!] vdbs 폴더에 유효한 벡터 DB가 없습니다. 프로그램을 종료합니다.")
    exit()

# === 벡터 DB 통합 ===
print("[*] 벡터 DB 통합 중...")
if not indexes:
     print("[!] 로드된 인덱스가 없습니다.")
     exit()

try:
    dim = indexes[0].d
    merged_index = faiss.IndexFlatL2(dim)
    chunk_metadata = [] # 각 chunk의 원본 텍스트와 출처 저장

    current_offset = 0
    for i, (idx, ch) in enumerate(zip(indexes, chunks_all)):
        if idx.d != dim:
            print(f"   - [경고] 인덱스 {i}의 차원({idx.d})이 첫 번째 인덱스({dim})와 다릅니다. 건너<0xEB><0><0x8E><0xB0>니다.")
            continue

        try:
            vecs = idx.reconstruct_n(0, idx.ntotal)
            merged_index.add(vecs)
            source_list = doc_sources[current_offset : current_offset + len(ch)]
            chunk_metadata.extend([{"text": text, "source": source} for text, source in zip(ch, source_list)])
            current_offset += len(ch)

        except Exception as inner_e:
             print(f"   - [오류] 인덱스 {i} 처리 중 오류: {inner_e}")


    if merged_index.ntotal == 0:
        print("[!] 통합할 유효한 벡터 DB가 없습니다. 프로그램을 종료합니다.")
        exit()

    print(f"[*] 벡터 DB 통합 완료. 총 문단 수: {merged_index.ntotal}")

except Exception as e:
     print(f"[!] 벡터 DB 통합 중 오류 발생: {e}")
     exit()


print("\n[*] RAG 챗봇 시작. 종료하려면 'exit' 입력.")
# === RAG 루프 ===
while True:
    query = input("\n[User] >> ").strip()
    if query.lower() in ["exit", "quit", "종료"]:
        print("[*] 챗봇을 종료합니다.")
        break
    if not query:
        continue

    try:
        # 1. 질문 임베딩
        q_emb = embedder.encode([query])

        # 2. 유사도 검색 (k=3 -> 상위 3개 결과)
        distances, indices = merged_index.search(q_emb, k=3)

        # 3. 검색된 문서 조각 및 출처 가져오기
        retrieved_docs_text = [] # ★ 수정 ★ 실제 텍스트 저장용
        retrieved_docs_info = [] # ★ 추가 ★ 텍스트와 출처 함께 저장용 (원본 출력 위해)
        context_sources = set() # 중복 출처 제거용 (참고용으로 유지)

        for i in indices[0]:
            if 0 <= i < len(chunk_metadata): # 유효한 인덱스인지 확인
                doc_info = chunk_metadata[i]
                retrieved_docs_text.append(doc_info["text"]) # 프롬프트용 텍스트
                retrieved_docs_info.append(doc_info)       # 원본 출력용 정보
                context_sources.add(doc_info["source"])    # 파일명 추적 (선택 사항)
            else:
                print(f"   - [경고] 잘못된 인덱스 {i}가 검색되었습니다.")


        if not retrieved_docs_text:
            print("\n[AI] : 죄송합니다. 관련 정보를 찾을 수 없습니다.")
            continue

        context = "\n\n---\n\n".join(retrieved_docs_text) # API 전달용 Context

        # 4. 프롬프트 생성
        prompt = f"""
다음은 검색된 관련 문서 내용입니다:
---
{context}
---

위 내용을 요약해줘.
"""
#위 내용을 참고하여 아래 질문에 대해 한국어로 명확히 답변해주세요. 내용을 찾을 수 없다면 "정보를 찾을 수 없습니다." 라고 답해주세요.:
#{query}

        # 5. Gemini API 호출 및 시간 측정
        start_api_time = time.time()
        try:
            safety_settings = [ # Safety Settings 추가
                {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_NONE"},
                {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_NONE"},
                {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_NONE"},
                {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_NONE"},
            ]
            response = model.generate_content(prompt, safety_settings=safety_settings)

            if not response.parts:
                 if response.prompt_feedback and response.prompt_feedback.block_reason:
                     ai_response = f"[!] API 응답 차단됨: {response.prompt_feedback.block_reason}"
                 else:
                     ai_response = "[!] API로부터 빈 응답을 받았습니다."
            else:
                 ai_response = response.text.strip()

        except Exception as api_error:
            print(f"\n[!] Gemini API 호출 중 오류 발생: {api_error}")
            ai_response = "[!] 답변 생성 중 오류가 발생했습니다."

        end_api_time = time.time()
        response_time = end_api_time - start_api_time

        # 6. 결과 출력 ★★★ 수정된 부분 ★★★
        print("\n[AI] : ", ai_response)
        print("-" * 30)
        print("--- 참고 원문 (Context) ---")
        for idx, doc_info in enumerate(retrieved_docs_info):
            print(f"\n[출처: {doc_info['source']} / 조각 {idx+1}]")
            print(doc_info['text']) # 실제 텍스트 조각 내용 출력
        print("-" * 30) # 원문 끝 구분선
        print(f"응답 시간: {response_time:.2f} 초")
        print("-" * 30)
        # ★★★ 수정 끝 ★★★

    except Exception as e:
        print(f"\n[!] 처리 중 오류 발생: {e}")