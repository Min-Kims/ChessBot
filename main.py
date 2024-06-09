import os
import streamlit as st
from langchain.embeddings import CacheBackedEmbeddings
from langchain.storage import LocalFileStore
from langchain_openai import OpenAIEmbeddings
from langchain_community.embeddings.huggingface import HuggingFaceEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_core.messages import ChatMessage
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders.unstructured import UnstructuredFileLoader
from langchain_community.vectorstores.faiss import FAISS
from langserve import RemoteRunnable
from langchain_openai import ChatOpenAI
from langchain_core.callbacks.streaming_stdout import StreamingStdOutCallbackHandler


# ⭐️ Embedding 설정
# USE_BGE_EMBEDDING = True 로 설정시 HuggingFace BAAI/bge-m3 임베딩 사용 (2.7GB 다운로드 시간 걸릴 수 있습니다)
# USE_BGE_EMBEDDING = False 로 설정시 OpenAIEmbeddings 사용 (OPENAI_API_KEY 입력 필요. 과금)
USE_BGE_EMBEDDING = True

if not USE_BGE_EMBEDDING:
    # OPENAI API KEY 입력
    # Embedding 을 무료 한글 임베딩으로 대체하면 필요 없음!
    os.environ["OPENAI_API_KEY"] = "OPENAI API KEY 입력"

# ⭐️ LangServe 모델 설정(EndPoint)
# 1) REMOTE 접속: 본인의 REMOTE LANGSERVE 주소 입력
# (예시)
#LANGSERVE_ENDPOINT = "https://poodle-deep-marmot.ngrok-free.app/llm/"
#민 예시 LANGSERVE_ENDPOINT = "https://8081-182-212-208-243.ngrok-free.app/llm/"

# 2) LocalHost 접속: 끝에 붙는 N4XyA 는 각자 다르니
# http://localhost:8000/llm/playground 에서 python SDK 에서 확인!
LANGSERVE_ENDPOINT = "http://localhost:8000/llm/c/N4XyA"
#김민 http://localhost:8000/xionic/c/N4XyA

# 필수 디렉토리 생성 @Mineru
if not os.path.exists(".cache"):
    os.mkdir(".cache")
if not os.path.exists(".cache/embeddings"):
    os.mkdir(".cache/embeddings")
if not os.path.exists(".cache/files"):
    os.mkdir(".cache/files")

# 프롬프트를 자유롭게 수정해 보세요!
RAG_PROMPT_TEMPLATE = """당신은 체스 전문가입니다. 체스 선생님처럼 답변하세요. 모든 대답은 한국어로 하고, 질문에 친절하게 답변하는 AI 입니다. 답을 모른다면 모른다고 답변하세요.  
Question: {question} 
Context: {context} 
Answer:"""

st.set_page_config(
    page_title="♟️Chess Chatbot♟️",
    page_icon="♟️"
)

st.markdown("""
<style>
            
img {
    max-heigh: 300px;
}
.streamlit-expanderContent div {
    display : flex;
    justify-content: center;
    font-size: 20px;
            
.st-emotion-cache-j6qv4b e1nzilvr5 div {
    font-size: 10px;
}
}
[data-testid="stExpanderToggleIcon"] {
    visibility: hidden;
}      
</style>    
""", unsafe_allow_html=True)


st.title("Chess GPT♟️")
st.markdown("Made by. KM")

#사진 추가
picture_path = "C:/Users/aasxs/OneDrive/바탕 화면/사진/DALL·E 2024-05-23 23.27.44 - A high-tech chessboard with chrome and glass pieces poised for a game, viewed from a side angle. The background is a high-tech laboratory with robotic.webp"

# expander : 접기펼치기, label = 이름, expanded=True : 기본적으로 펼치기
# with 하위에 넣은 것들이 전부 펼치기, 접기에 포함
with st.expander(label = "Chess Image" ,expanded=True):
    st.image(picture_path)

#체스 선수들 컬럼별로 사진 추가하기
initial_grandmaster = [
    {
        "name" : "Magnus Carlsen",
        "country" : "Norway",
        "rating" : 2830,
        "image_url" : "https://images.chesscomfiles.com/uploads/v1/master_player/0541e1fa-ad51-11eb-8ff2-ef206b1039dc.28e1af01.250x250o.854dcaef7310@2x.jpg"
    },
    {
        "name" : "Hikaru",
        "country" : "USA",
        "rating" : 2794,
        "image_url" : "https://images.chesscomfiles.com/uploads/v1/master_player/c9d9e712-b795-11eb-ad09-a98e159bb142.71b234af.250x250o.8ec3b4a5d8bb@2x.jpeg"
    },
    {
        "name" : "Fabiano Caruana",
        "country" : "USA",
        "rating" : 2805,
        "image_url" : "https://images.chesscomfiles.com/uploads/v1/master_player/c69f503e-9b37-11eb-9e62-fb361d491281.d47636d1.250x250o.d8a2bc99d402@2x.jpeg"
    },
    {
        "name" : "Ian Nepomniachtchi",
        "country" : "Rusia",
        "rating" : 2770,
        "image_url" : "https://images.chesscomfiles.com/uploads/v1/master_player/9717c30a-ae0b-11eb-8a19-935f50840bf3.88bdb385.250x250o.0ecb6f67541f@2x.png"
    },
    {
        "name" : "Nodirbek Abdusattorov",
        "country" : "Uzbekistan",
        "rating" : 2766,
        "image_url" : "https://images.chesscomfiles.com/uploads/v1/master_player/bdf65d10-6151-11ed-b155-8b0a0660e799.ae11d334.250x250o.f4e88889cf1e@2x.jpg"
    },
    {
        "name" : "Gukesh Dommaraju",
        "country" : "India",
        "rating" : 2763,
        "image_url" : "https://images.chesscomfiles.com/uploads/v1/master_player/918644b8-e1b9-11ee-9575-f74f04d3e9fc.a89af8fc.250x250o.6164fe81dcb6@2x.png"
    },
    {
        "name" : "Hanrim Kang",
        "country" : "South Korea",
        "rating" : 1400,
        "image_url" : "https://scontent-ssn1-1.xx.fbcdn.net/v/t1.18169-9/15781217_610282815822658_68050988425839739_n.jpg?_nc_cat=110&ccb=1-7&_nc_sid=5f2048&_nc_ohc=CZRLXZe7uEYQ7kNvgGdY7cV&_nc_ht=scontent-ssn1-1.xx&oh=00_AYA1O1cb0cXC7_Dpg0oN7RaZbaBtaNLff5V2clnVbepV5g&oe=6676BEEE"
    },
    {
        "name" : "Min Kim",
        "country" : "South Korea",
        "rating" : 1120,
        "image_url" : "C:/Users/aasxs/OneDrive/바탕 화면/사진/kimmin_profiles.jpg"
    }
    
]

if "grandmaster" not in st.session_state:
    st.session_state.grandmaster = initial_grandmaster

#나라 등록
country_dict = {
    "USA" : "💵",
    "Korea" : ":kr:",
    "Norway" : "norway",
    "India" : "india",
    "Uzbekistan" : "😊",
    "Rusia" : "😂"
}

#선수등록 
# with st.form(key="form"):
#     col1, col2, col3 = st.columns(3)
#     with col1:
#         name = st.text_input(label="Player Name")
#     with col2:
#         countries = st.multiselect(
#             label="Country", 
#             options=list(country_dict.keys()),
#             max_selections= 2
#         )
#     with col3:
#         ratings = st.text_input(label="Rating")
#     image_url = st.text_input(label="Player Image")
#     submit = st.form_submit_button(label= "Submit")
#     if submit:
#         if not name:
#             st.error("Player이름을 입력해주세요.")
#         elif len(countries) == 0:
#             st.error("나라를 적어도 1개 이상 입력해주세요.")
#         elif not ratings:
#             st.error("rating을 입력해주세요.")
#         else:
#             st.success("선수등록을 성공하였습니다.")
#             grandmaster.append({
#                 "name" : name,
#                 "country" : countries,
#                 "rating" : ratings,
#                 "image_url" : image_url if image_url else "./images/default.png"
#             })


#사이드바 설정
with st.sidebar:
    file = st.file_uploader(
        "기보 업로드",
        type=["pdf", "txt", "docx"],
    )
    #Test해본 것 
    with st.form(key="form"):
        col1, col2, col3 = st.columns(3)
        with col1:
            name = st.text_input(label="Player Name")
        with col2:
            countries = st.multiselect(
                label="Country", 
                options=list(country_dict.keys()),
                max_selections= 2
            )
        with col3:
            ratings = st.text_input(label="Rating")
        image_url = st.text_input(label="Player Image")
        submit = st.form_submit_button(label= "Submit")
        if submit:
            if not name:
                st.error("Player이름을 입력해주세요.")
            elif len(countries) == 0:
                st.error("나라를 적어도 1개 이상 입력해주세요.")
            elif not ratings:
                st.error("rating을 입력해주세요.")
            else:
                st.success("선수등록을 성공하였습니다.")
                initial_grandmaster.append({
                    "name" : name,
                    "country" : countries,
                    "rating" : ratings,
                    "image_url" : image_url if image_url else "./images/default.png"
                })


#이미지 배치 (선수들) 
for i in range(0, len(st.session_state.grandmaster), 3):
    row_grandmaster = st.session_state.grandmaster[i:i+3]
    cols = st.columns(3)
    for j in range(len(row_grandmaster)):
        with cols[j]:
            player = row_grandmaster[j]
            with st.expander(label=f"**{i+j+1}.{player["name"]}**", expanded=True):
                st.image(player["image_url"])
                #삭제버튼 
                delete_button = st.button(label="Delete", key=i+j, use_container_width=True)
                if delete_button:
                    del st.session_state.grandmaster[i+j]
                    st.rerun()


#####Logic 관련#####


if "messages" not in st.session_state:
    st.session_state["messages"] = [
        ChatMessage(role="assistant", content="무엇을 도와드릴까요?")
    ]


def print_history():
    for msg in st.session_state.messages:
        st.chat_message(msg.role).write(msg.content)


def add_history(role, content):
    st.session_state.messages.append(ChatMessage(role=role, content=content))


def format_docs(docs):
    # 검색한 문서 결과를 하나의 문단으로 합쳐줍니다.
    return "\n\n".join(doc.page_content for doc in docs)


@st.cache_resource(show_spinner="Embedding file...")
def embed_file(file):
    file_content = file.read()
    file_path = f"./.cache/files/{file.name}"
    with open(file_path, "wb") as f:
        f.write(file_content)

    cache_dir = LocalFileStore(f"./.cache/embeddings/{file.name}")

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50,
        separators=["\n\n", "\n", "(?<=\. )", " ", ""],
        length_function=len,
    )
    loader = UnstructuredFileLoader(file_path)
    docs = loader.load_and_split(text_splitter=text_splitter)

    if USE_BGE_EMBEDDING:
        # BGE Embedding: @Mineru
        model_name = "BAAI/bge-m3"
        # GPU Device 설정:
        # - NVidia GPU: "cuda"
        # - Mac M1, M2, M3: "mps"
        # - CPU: "cpu"
        model_kwargs = {
            # "device": "cuda"
            # "device": "mps"   #Default
            "device": "cpu"    #김민 수정했음
        }
        encode_kwargs = {"normalize_embeddings": True}
        embeddings = HuggingFaceEmbeddings(
            model_name=model_name,
            model_kwargs=model_kwargs,
            encode_kwargs=encode_kwargs,
        )
    else:
        embeddings = OpenAIEmbeddings()
    cached_embeddings = CacheBackedEmbeddings.from_bytes_store(embeddings, cache_dir)
    vectorstore = FAISS.from_documents(docs, embedding=cached_embeddings)
    retriever = vectorstore.as_retriever()
    return retriever


def format_docs(docs):
    # 검색한 문서 결과를 하나의 문단으로 합쳐줍니다.
    return "\n\n".join(doc.page_content for doc in docs)








if file:
    retriever = embed_file(file)

print_history()


if user_input := st.chat_input():
    add_history("user", user_input)
    st.chat_message("user").write(user_input)
    with st.chat_message("assistant"):
        # ngrok remote 주소 설정
        ollama = RemoteRunnable(LANGSERVE_ENDPOINT)
        # LM Studio 모델 설정
        # ollama = ChatOpenAI(
        #     base_url="http://localhost:1234/v1",
        #     api_key="lm-studio",
        #     model="teddylee777/EEVE-Korean-Instruct-10.8B-v1.0-gguf",
        #     streaming=True,
        #     callbacks=[StreamingStdOutCallbackHandler()],  # 스트리밍 콜백 추가
        # )
        chat_container = st.empty()
        if file is not None:
            prompt = ChatPromptTemplate.from_template(RAG_PROMPT_TEMPLATE)

            # 체인을 생성합니다.
            rag_chain = (
                {
                    "context": retriever | format_docs,
                    "question": RunnablePassthrough(),
                }
                | prompt
                | ollama
                | StrOutputParser()
            )
            # 문서에 대한 질의를 입력하고, 답변을 출력합니다.
            answer = rag_chain.stream(user_input)  # 문서에 대한 질의
            chunks = []
            for chunk in answer:
                chunks.append(chunk)
                chat_container.markdown("".join(chunks))
            add_history("ai", "".join(chunks))
        else:
            prompt = ChatPromptTemplate.from_template(
                "다음의 질문에 간결하게 답변해 주세요:\n{input}"
            )

            # 체인을 생성합니다.
            chain = prompt | ollama | StrOutputParser()

            answer = chain.stream(user_input)  # 문서에 대한 질의
            chunks = []
            for chunk in answer:
                chunks.append(chunk)
                chat_container.markdown("".join(chunks))
            add_history("ai", "".join(chunks))
