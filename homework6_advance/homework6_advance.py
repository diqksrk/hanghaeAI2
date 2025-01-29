import os
import base64
import streamlit as st
from dotenv import load_dotenv
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.embeddings import OpenAIEmbeddings
from langchain_openai import ChatOpenAI
from PyPDF2 import PdfReader
import requests
import shutil
from io import BytesIO

st.set_page_config(page_title="논문 요약 서비스", layout="centered")

st.title("논문 요약 서비스")
# ChromaDB 데이터 저장 경로
CHROMA_PERSIST_DIR = "./chroma_db"

# 새로운 세션이 시작될 때 기존 벡터 DB 삭제
if "initialized" not in st.session_state:
    if os.path.exists(CHROMA_PERSIST_DIR):
        shutil.rmtree(CHROMA_PERSIST_DIR)  # 기존 데이터 삭제
    st.session_state.initialized = True
    st.session_state.vectorstore = None  # 새 세션이므로 벡터스토어 초기화

# GPT 모델 설정
llm = ChatOpenAI(model="gpt-4o-mini", api_key="API_KEY")

if "pdf_entries" not in st.session_state:
    st.session_state.pdf_entries = []
    st.session_state.file_names = []
    st.session_state.messages = []

# PDF 업로드 또는 URL 입력
st.subheader("논문을 업로드하거나 URL을 입력하세요")
pdf_source = st.radio("선택하세요:", ("파일 업로드", "URL 입력"), key="pdf_source")

if pdf_source == "파일 업로드":
    uploaded_file = st.file_uploader("PDF 파일을 업로드하세요", type=['pdf'], key="file_uploader")
    if uploaded_file:
        st.session_state.pdf_entries.append(BytesIO(uploaded_file.read()))
        st.session_state.file_names.append(uploaded_file.name)
        st.success(f"파일 '{uploaded_file.name}' 업로드 완료")

elif pdf_source == "URL 입력":
    pdf_url = st.text_input("PDF 파일의 URL을 입력하세요")
    if pdf_url and st.button("URL에서 가져오기"):
        try:
            response = requests.get(pdf_url)
            if response.status_code == 200:
                st.session_state.pdf_entries.append(BytesIO(response.content))
                st.session_state.file_names.append(pdf_url)
                st.success("PDF 다운로드 성공")
            else:
                st.error("PDF를 다운로드할 수 없습니다. URL을 확인하세요.")
        except Exception as e:
            st.error(f"오류 발생: {str(e)}")

# 현재 업로드된 파일 목록
if st.session_state.file_names:
    st.subheader("업로드된 논문 목록")
    for idx, file_name in enumerate(st.session_state.file_names):
        st.write(f"{idx + 1}. {file_name}")

# 대화 기록 출력
st.subheader("대화 기록")
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# 질문 입력 창
st.divider()
user_input = st.chat_input("질문을 입력하세요:")
if user_input:
    with st.chat_message("user"):
        st.markdown(user_input)
    st.session_state.messages.append({"role": "user", "content": user_input})

    if st.session_state.pdf_entries:
        try:
            all_pages = []
            for pdf_stream in st.session_state.pdf_entries:
                reader = PdfReader(pdf_stream)
                pages = [page.extract_text() if page.extract_text() else "" for page in reader.pages]

                def filter_references(pages):
                    """References 이후의 내용을 제거한 페이지 리스트 반환"""
                    filtered_pages = []
                    for page in pages:
                        if "References" in page:
                            break
                        filtered_pages.append(page)
                    return filtered_pages

                all_pages.extend(filter_references(pages))

            if not any(all_pages):
                st.error("PDF에서 텍스트를 추출할 수 없습니다. 다른 파일을 사용해 보세요.")
                st.stop()

            # 텍스트 분할
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000,
                chunk_overlap=200
            )
            documents = text_splitter.create_documents(all_pages)

            if not documents:
                st.error("텍스트 분할에 실패했습니다. PDF 파일의 내용을 확인하세요.")
                st.stop()

            # 벡터 스토어 생성
            vectorstore = Chroma.from_documents(
                documents=documents,
                embedding=OpenAIEmbeddings(api_key="API_KEY"),
                persist_directory=CHROMA_PERSIST_DIR  # 데이터 저장
            )

            retriever = vectorstore.as_retriever()

            # 프롬프트 확장: 요약 관련 키워드 감지
            if any(kw in user_input.lower() for kw in ["요약", "summary", "idea", "결론", "향후"]):
                query = f"""
                다음과 같은 순서로 요약 해줘.
                1. 주요 아이디어
                2. 실험 및 결과
                3. 결론 
                4. 향후 방향성 및 기대효과
                질문: {user_input}
                """
            else:
                query = user_input

            retrieved_docs = retriever.invoke(query)
            retrieved_text = " ".join([doc.page_content for doc in retrieved_docs])

            if not retrieved_text.strip():
                st.error("검색된 내용이 없습니다. PDF의 내용을 확인해주세요.")
            else:
                if "요약" in query:
                    summary_prompt = f"""
                    내용: {retrieved_text[:3000]}
                    
                    다음과 같은 순서로 요약 해줘.
                    1. 주요 아이디어
                    - 핵심 아이디어를 특히 강조해줘.
                    2. 실험 및 결과
                    - 실험 및 결과는 중요한 부분이니 자세히 요약해줘.
                    3. 결론
                    4. 향후 방향성 및 기대효과
                    
                    특히, 각 항목별 요약은 3-5 문장으로 요약해줘. 각 줄별로 줄바꿈 처리를 해줘
                    해당 양식을 지키지 않을시에는 큰 벌점이 주어질거야.
                    """

                    summary = llm(summary_prompt)

                    with st.chat_message("assistant"):
                        st.markdown(summary.content)
                    st.session_state.messages.append({"role": "assistant", "content": summary.content})
                else:
                    summary_prompt = f"""
                    내용: {retrieved_text[:1500]}
                    
                    질문 : {user_input}
                    """
                    summary = llm(summary_prompt)

                    with st.chat_message("assistant"):
                        st.markdown(summary.content)
                    st.session_state.messages.append({"role": "assistant", "content": summary.content})
        except Exception as e:
            st.error(f"오류 발생: {str(e)}")
    else:
        st.error("논문이 업로드되지 않았습니다. 파일을 업로드하거나 URL을 입력하세요.")
