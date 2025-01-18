import base64
import streamlit as st
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage

st.title("Fashion Recommendation Bot")

# GPT 모델 설정
model = ChatOpenAI(model="gpt-4o-mini", api_key="API_KEY")

# Session state 초기화
if "images" not in st.session_state:
    st.session_state.images = []
if "messages" not in st.session_state:
    st.session_state.messages = []

# 여러 이미지를 업로드할 수 있도록 수정
uploaded_files = st.file_uploader("본인의 전신이 보이는 사진을 올려주세요!", type=['png', 'jpg', 'jpeg'], accept_multiple_files=True)

if uploaded_files:
    st.session_state.images = []
    for uploaded_file in uploaded_files:
        st.image(uploaded_file)
        image = base64.b64encode(uploaded_file.read()).decode("utf-8")
        st.session_state.images.append(f"data:image/jpeg;base64,{image}")

# 이전 메시지들을 UI에 띄우기
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# 사용자로부터 질문을 입력받기
if user_question := st.chat_input("질문을 입력하세요:"):
    with st.chat_message("user"):
        st.markdown(user_question)
    st.session_state.messages.append({"role": "user", "content": user_question})

    # 메시지 생성
    message_content = [
        {"type": "text", "text": "다음은 여러 장의 사진입니다. 각 사진에 대해 질문에 답변해주세요."},
    ]
    for image in st.session_state.images:
        message_content.append({"type": "image_url", "image_url": {"url": image}})
    message_content.append({"type": "text", "text": user_question})

    message = HumanMessage(content=message_content)

    # GPT에게 질문 전달 및 답변 받기
    result = model.invoke([message])
    response = result.content

    with st.chat_message("assistant"):
        st.markdown(response)
    st.session_state.messages.append({"role": "assistant", "content": response})
