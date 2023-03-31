import os

import openai
import streamlit as st
from streamlit_chat import message

from langchain.chat_models import ChatOpenAI
from langchain.prompts.chat import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
    MessagesPlaceholder,
)
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationChain
from langchain.callbacks.base import CallbackManager

# from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.callbacks.streamlit import StreamlitCallbackHandler

from dotenv import load_dotenv

load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

system_message = """
あなたはエンジニアリングの専門領域について高度な知識を持つアシスタントです。ユーザはエンジニアで、あなたにプログラミングやプロダクト開発に関する質問を投げかけます。
アシスタントとして、プロダクト開発に役立つ回答を、できる限り根拠やリファレンスのURLを示した上で返してください。リファレンスは公式のドキュメントであるほど望ましいです。"""

prompt = ChatPromptTemplate.from_messages(
    [
        SystemMessagePromptTemplate.from_template(system_message),
        MessagesPlaceholder(variable_name="history"),
        HumanMessagePromptTemplate.from_template("{input}"),
    ]
)


@st.cache_resource
def load_conversation():
    llm = ChatOpenAI(
        streaming=True,
        callback_manager=CallbackManager(
            [StreamlitCallbackHandler()]
        ),  # if adding StreamingStdOutCallbackHandler, output the responses in StdOut
        verbose=False,
        temperature=0,
        max_tokens=1024,
    )
    memory = ConversationBufferMemory(return_messages=True)
    conversation = ConversationChain(memory=memory, prompt=prompt, llm=llm)
    return conversation


# Run ojisan
st.title("エンジニアリングおじさん")

if "generated" not in st.session_state:
    st.session_state.generated = []
if "past" not in st.session_state:
    st.session_state.past = []


with st.form("おじさんに質問する", clear_on_submit=True):
    user_message = st.text_area("質問を入力してください")
    submitted = st.form_submit_button("質問する")

    if submitted:
        conversation = load_conversation()
        answer = conversation.predict(input=user_message)
        st.session_state.past.append(user_message)
        st.session_state.generated.append(answer)

        if st.session_state["generated"]:
            for i in range(len(st.session_state.generated) - 1, -1, -1):
                message(st.session_state.generated[i], key=str(i))
                message(st.session_state.past[i], is_user=True, key=str(i) + "_user")
