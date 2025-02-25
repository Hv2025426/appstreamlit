import os
import asyncio
import time
from datetime import datetime
import pandas as pd
import tiktoken
from io import BytesIO
import plotly.express as px
from dotenv import load_dotenv
from openai import OpenAI
import anthropic
from groq import Groq
import google.generativeai as genai
import streamlit as st

# Configuração da página
st.set_page_config(page_title="Recurso Aprovado - Ambiente de Validação", layout="wide")

# Carregar variáveis de ambiente
load_dotenv()

# Inicialização de clientes para diferentes serviços de IA
@st.cache_resource
def init_clients():
    """Inicializa clientes para diferentes serviços de IA"""
    genai.configure(api_key=os.getenv('GOOGLE_API_KEY'))
    return {
        'openai': OpenAI(api_key=os.getenv('OPENAI_API_KEY')),
        'openrouter': OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=os.getenv('OPENROUTER_API_KEY'),  # Substitua pela sua chave real da Openrouter
            
        ),
        'claude': anthropic.Anthropic(api_key=os.getenv('ANTHROPIC_API_KEY')),
        'maritalk': OpenAI(
            api_key=os.getenv('MARITALK_API_KEY'),
            base_url="https://chat.maritaca.ai/api"
        ),
        'nebius_405b': OpenAI(
            api_key=os.getenv('NEBIUS_API_KEY'),
            base_url="https://api.studio.nebius.ai/v1/"
        ),
        'nebius_70b': OpenAI(
            api_key=os.getenv('NEBIUS_API_KEY'),
            base_url="https://api.studio.nebius.ai/v1/"
        ),
        'groq': Groq(api_key=os.getenv('GROQ_API_KEY')),
        'deepseek': OpenAI(
            api_key=os.getenv('DEEPSEEK_API_KEY'),
            base_url="https://api.deepseek.com"
        ),
        'gemini': genai
    }

# Tabela de custo por token (em dólares) - (custo de input, custo de output)
COSTS = {
    'gpt': (0.15, 0.60),
    'gpt4': (2.50, 10.00),
    'claude': (3.00, 15.00),
    'maritalk': (0.83, 1.67),
    'llama405b': (1.00, 3.00),
    'llama70b_nebius': (0.13, 0.40),
    'llama70b_groq': (0.59, 0.79),
    'deepseek_chat': (0.27, 1.10),
    'deepseek_reasoner': (0.55, 2.19),
    'gemini': (0.10, 0.40),
    'o3-mini-high': (1.1, 4.4),
    'qwen-max': (1.6, 6.4),
    'qwen-plus': (0.4, 1.2)
}

# Funções de análise para os modelos utilizando bibliotecas nativas (mantidas sem alterações)
async def analyze_with_gpt(messages):
    start_time = time.time()
    try:
        system_message = [{"role": "system", "content": "Você é um assistente inteligente"}]
        full_messages = system_message + messages
        response = await asyncio.to_thread(
            st.session_state.clients['openai'].chat.completions.create,
            model="gpt-4o-mini",
            messages=full_messages,
            temperature=0.5
        )
        ai_response = response.choices[0].message.content
        return ai_response, response.usage.prompt_tokens, response.usage.completion_tokens, time.time() - start_time
    except Exception as e:
        st.error(f"Erro no GPT-4o-Mini: {str(e)}")
        return None, 0, 0, 0.0

async def analyze_with_gpt4(messages):
    start_time = time.time()
    try:
        system_message = [{"role": "system", "content": "Você é um assistente inteligente"}]
        full_messages = system_message + messages
        response = await asyncio.to_thread(
            st.session_state.clients['openai'].chat.completions.create,
            model="gpt-4",
            messages=full_messages,
            temperature=0.5
        )
        ai_response = response.choices[0].message.content
        return ai_response, response.usage.prompt_tokens, response.usage.completion_tokens, time.time() - start_time
    except Exception as e:
        st.error(f"Erro no GPT-4: {str(e)}")
        return None, 0, 0, 0.0

async def analyze_with_claude(messages):
    start_time = time.time()
    try:
        response = await asyncio.to_thread(
            st.session_state.clients['claude'].messages.create,
            model="claude-3-7-sonnet-20250219",
            max_tokens=4000,
            temperature=0.5,
            messages=messages,
            system="Você é um assistente inteligente"
        )
        ai_response = response.content[0].text
        return ai_response, response.usage.input_tokens, response.usage.output_tokens, time.time() - start_time
    except Exception as e:
        st.error(f"Erro no Claude: {str(e)}")
        return None, 0, 0, 0.0

async def analyze_maritalk(messages):
    start_time = time.time()
    try:
        response = await asyncio.to_thread(
            st.session_state.clients['maritalk'].chat.completions.create,
            model="sabia-3",
            messages=messages,
            max_tokens=8000,
            temperature=0.7
        )
        ai_response = response.choices[0].message.content
        encoding = tiktoken.get_encoding("cl100k_base")
        input_tokens = sum(len(encoding.encode(m["content"])) for m in messages)
        output_tokens = len(encoding.encode(ai_response))
        return ai_response, input_tokens, output_tokens, time.time() - start_time
    except Exception as e:
        st.error(f"Erro no Maritalk: {str(e)}")
        return None, 0, 0, 0.0

async def analyze_llama405b(messages):
    start_time = time.time()
    try:
        response = await asyncio.to_thread(
            st.session_state.clients['nebius_405b'].chat.completions.create,
            model="meta-llama/Meta-Llama-3.1-405B-Instruct",
            messages=messages,
            max_tokens=4096,
            temperature=0.6
        )
        ai_response = response.choices[0].message.content
        encoding = tiktoken.get_encoding("cl100k_base")
        input_tokens = sum(len(encoding.encode(m["content"])) for m in messages)
        output_tokens = len(encoding.encode(ai_response))
        return ai_response, input_tokens, output_tokens, time.time() - start_time
    except Exception as e:
        st.error(f"Erro no Llama 405B: {str(e)}")
        return None, 0, 0, 0.0

async def analyze_llama70b_nebius(messages):
    start_time = time.time()
    try:
        response = await asyncio.to_thread(
            st.session_state.clients['nebius_70b'].chat.completions.create,
            model="meta-llama/Llama-3.3-70B-Instruct",
            messages=messages,
            max_tokens=20000,
            temperature=0.6
        )
        ai_response = response.choices[0].message.content
        encoding = tiktoken.get_encoding("cl100k_base")
        input_tokens = sum(len(encoding.encode(m["content"])) for m in messages)
        output_tokens = len(encoding.encode(ai_response))
        return ai_response, input_tokens, output_tokens, time.time() - start_time
    except Exception as e:
        st.error(f"Erro no Llama 70B (Nebius): {str(e)}")
        return None, 0, 0, 0.0

async def analyze_llama70b_groq(messages):
    start_time = time.time()
    try:
        response = await asyncio.to_thread(
            st.session_state.clients['groq'].chat.completions.create,
            model="llama-3.3-70b-versatile",
            messages=messages,
            temperature=0.6,
            max_tokens=20000
        )
        ai_response = response.choices[0].message.content
        encoding = tiktoken.get_encoding("cl100k_base")
        input_tokens = sum(len(encoding.encode(m["content"])) for m in messages)
        output_tokens = len(encoding.encode(ai_response))
        return ai_response, input_tokens, output_tokens, time.time() - start_time
    except Exception as e:
        st.error(f"Erro no Llama 70B (Groq): {str(e)}")
        return None, 0, 0, 0.0

async def analyze_deepseek_chat(messages):
    start_time = time.time()
    try:
        response = await asyncio.to_thread(
            st.session_state.clients['deepseek'].chat.completions.create,
            model="deepseek-chat",
            messages=messages,
            max_tokens=8192,
            temperature=0.7
        )
        ai_response = response.choices[0].message.content
        encoding = tiktoken.get_encoding("cl100k_base")
        input_tokens = sum(len(encoding.encode(m["content"])) for m in messages)
        output_tokens = len(encoding.encode(ai_response))
        return ai_response, input_tokens, output_tokens, time.time() - start_time
    except Exception as e:
        st.error(f"Erro no Deepseek Chat: {str(e)}")
        return None, 0, 0, 0.0

async def analyze_deepseek_reasoner(messages):
    start_time = time.time()
    try:
        formatted_messages = []
        system_added = False
        for msg in messages:
            if msg["role"] == "system":
                formatted_messages.append(msg)
                system_added = True
            else:
                if len(formatted_messages) > 0 and formatted_messages[-1]["role"] == msg["role"]:
                    formatted_messages.append({"role": "assistant" if msg["role"] == "user" else "user", "content": "[Continuação]"})
                formatted_messages.append(msg)
        if not system_added:
            formatted_messages.insert(0, {"role": "system", "content": "Você é um assistente inteligente"})
        response = await asyncio.to_thread(
            st.session_state.clients['deepseek'].chat.completions.create,
            model="deepseek-reasoner",
            messages=formatted_messages,
            max_tokens=8192,
            temperature=0.7
        )
        ai_response = response.choices[0].message.content
        encoding = tiktoken.get_encoding("cl100k_base")
        input_tokens = sum(len(encoding.encode(m["content"])) for m in formatted_messages)
        output_tokens = len(encoding.encode(ai_response))
        return ai_response, input_tokens, output_tokens, time.time() - start_time
    except Exception as e:
        st.error(f"Erro no Deepseek Reasoner: {str(e)}")
        return None, 0, 0, 0.0

async def analyze_with_gemini(messages):
    start_time = time.time()
    try:
        model = st.session_state.clients['gemini'].GenerativeModel("gemini-2.0-flash")
        user_messages = [m["content"] for m in messages if m["role"] == "user"]
        last_user_message = user_messages[-1] if user_messages else ""
        response = await asyncio.to_thread(
            model.generate_content,
            last_user_message
        )
        ai_response = response.text
        encoding = tiktoken.get_encoding("cl100k_base")
        input_tokens = sum(len(encoding.encode(m["content"])) for m in messages)
        output_tokens = len(encoding.encode(ai_response))
        return ai_response, input_tokens, output_tokens, time.time() - start_time
    except Exception as e:
        st.error(f"Erro no Gemini: {str(e)}")
        return None, 0, 0, 0.0

# Funções de análise para os novos modelos via Openrouter utilizando o OpenAI SDK
async def analyze_with_o3_mini_high(messages):
    start_time = time.time()
    try:
        system_message = [{"role": "system", "content": "Você é um assistente inteligente"}]
        full_messages = system_message + messages
        response = await asyncio.to_thread(
            st.session_state.clients['openrouter'].chat.completions.create,
            model="openai/o3-mini-high",
            messages=full_messages,
            temperature=0.5,
            extra_headers={
                "HTTP-Referer": os.getenv("YOUR_SITE_URL", ""),
                "X-Title": os.getenv("YOUR_SITE_NAME", "")
            }
        )
        ai_response = response.choices[0].message.content
        return ai_response, response.usage.prompt_tokens, response.usage.completion_tokens, time.time() - start_time
    except Exception as e:
        st.error(f"Erro no O3-Mini-high: {str(e)}")
        return None, 0, 0, 0.0

async def analyze_with_qwen_max(messages):
    start_time = time.time()
    try:
        system_message = [{"role": "system", "content": "Você é um assistente inteligente"}]
        full_messages = system_message + messages
        response = await asyncio.to_thread(
            st.session_state.clients['openrouter'].chat.completions.create,
            model="qwen/qwen-max",
            messages=full_messages,
            temperature=0.5,
            extra_headers={
                "HTTP-Referer": os.getenv("YOUR_SITE_URL", ""),
                "X-Title": os.getenv("YOUR_SITE_NAME", "")
            }
        )
        ai_response = response.choices[0].message.content
        return ai_response, response.usage.prompt_tokens, response.usage.completion_tokens, time.time() - start_time
    except Exception as e:
        st.error(f"Erro no Qwen-Max: {str(e)}")
        return None, 0, 0, 0.0

async def analyze_with_qwen_plus(messages):
    start_time = time.time()
    try:
        system_message = [{"role": "system", "content": "Você é um assistente inteligente"}]
        full_messages = system_message + messages
        response = await asyncio.to_thread(
            st.session_state.clients['openrouter'].chat.completions.create,
            model="qwen/qwen-plus",
            messages=full_messages,
            temperature=0.5,
            extra_headers={
                "HTTP-Referer": os.getenv("YOUR_SITE_URL", ""),
                "X-Title": os.getenv("YOUR_SITE_NAME", "")
            }
        )
        ai_response = response.choices[0].message.content
        return ai_response, response.usage.prompt_tokens, response.usage.completion_tokens, time.time() - start_time
    except Exception as e:
        st.error(f"Erro no Qwen-Plus: {str(e)}")
        return None, 0, 0, 0.0

# Função auxiliar para exibir mensagens com formatação
def display_message(role, content, model_name=None):
    if role == "user":
        st.markdown(
            f"<div style='background-color:#e6f7ff; padding:10px; margin-bottom:10px; border-radius:5px;'>"
            f"<strong>Você:</strong> {content}</div>",
            unsafe_allow_html=True
        )
    else:
        st.markdown(
            f"<div style='background-color:#f0f0f0; padding:10px; margin-bottom:10px; border-radius:5px;'>"
            f"<strong>{model_name}:</strong> {content}</div>",
            unsafe_allow_html=True
        )

# Interface de chat para cada modelo
def chat_interface(model_key, model_name, analysis_func):
    if f"{model_key}_messages" not in st.session_state:
        st.session_state[f"{model_key}_messages"] = []
    if f"{model_key}_ratings" not in st.session_state:
        st.session_state[f"{model_key}_ratings"] = []
    with st.expander(f"Histórico de Conversa ({model_name})", expanded=True):
        for msg in st.session_state[f"{model_key}_messages"]:
            if msg["role"] == "user":
                display_message("user", msg["content"])
            else:
                display_message("assistant", msg["content"], model_name)
    input_counter = st.session_state.get(f"{model_key}_input_counter", 0)
    user_input_key = f"{model_key}_input_{input_counter}"
    user_input = st.text_area(
        "Digite sua mensagem:", 
        key=user_input_key,
        height=180,
        placeholder=f"Escreva sua mensagem para {model_name}..."
    )
    col1, col2, col3, col4, col5 = st.columns([1.5, 2, 2, 2, 2])
    with col1:
        if st.button("Enviar Mensagem", key=f"send_{model_key}"):
            if user_input.strip():
                st.session_state[f"{model_key}_messages"].append({"role": "user", "content": user_input.strip()})
                st.session_state[f"{model_key}_rating_submitted"] = False
                st.session_state[f"{model_key}_input_counter"] = input_counter + 1
                async def process_message():
                    with st.spinner(f"{model_name} está pensando..."):
                        response, input_tokens, output_tokens, elapsed_time = await analysis_func(
                            st.session_state[f"{model_key}_messages"]
                        )
                    return response, input_tokens, output_tokens, elapsed_time
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                try:
                    response, input_tokens, output_tokens, elapsed_time = loop.run_until_complete(process_message())
                finally:
                    loop.close()
                if response:
                    st.session_state[f"{model_key}_messages"].append({"role": "assistant", "content": response})
                    st.session_state[f"{model_key}_last_input_tokens"] = input_tokens
                    st.session_state[f"{model_key}_last_output_tokens"] = output_tokens
                    st.session_state[f"{model_key}_last_time"] = elapsed_time
                    st.rerun()
    with col2:
        rating_counter = st.session_state.get(f"{model_key}_rating_counter", 0)
        rating_key = f"rating_{model_key}_{rating_counter}"
        rating = st.selectbox(
            "Avalie o modelo (1-5)",
            options=[1, 2, 3, 4, 5],
            index=0,
            key=rating_key,
            help="Selecione uma nota de 1 a 5 para avaliar o modelo."
        )
    with col3:
        if st.button("Enviar avaliação", key=f"rate_{model_key}"):
            if not st.session_state.get(f"{model_key}_rating_submitted", False):
                if 1 <= rating <= 5:
                    messages = st.session_state.get(f"{model_key}_messages", [])
                    last_user = next((msg["content"] for msg in reversed(messages) if msg["role"]=="user"), None)
                    last_assistant = next((msg["content"] for msg in reversed(messages) if msg["role"]=="assistant"), None)
                    if last_user and last_assistant:
                        input_tokens = st.session_state.get(f"{model_key}_last_input_tokens", 0)
                        output_tokens = st.session_state.get(f"{model_key}_last_output_tokens", 0)
                        elapsed_time = st.session_state.get(f"{model_key}_last_time", 0.0)
                        cost_input, cost_output = COSTS.get(model_key, (0.0, 0.0))
                        cost_usd = ((input_tokens * cost_input) + (output_tokens * cost_output)) / 1_000_000
                        cost_brl = cost_usd * 6
                        resultados = st.session_state.get("resultados", [])
                        resultados.append({
                            "Data": datetime.now().strftime("%d/%m/%Y"),
                            "modelo": model_name,
                            "prompt": last_user,
                            "resultado": last_assistant,
                            "Tempo": f"{elapsed_time:.2f} s",
                            "Input Tokens": input_tokens,
                            "Output Tokens": output_tokens,
                            "Custo (R$)": f"R${cost_brl:.4f}",
                            "nota obtida": rating
                        })
                        st.session_state["resultados"] = resultados
                        ratings = st.session_state[f"{model_key}_ratings"]
                        ratings.append(rating)
                        if len(ratings) > 10:
                            ratings.pop(0)
                        st.session_state[f"{model_key}_ratings"] = ratings
                        st.session_state[f"{model_key}_rating_submitted"] = True
                        st.success("Avaliação enviada com sucesso!")
                        st.rerun()
                    else:
                        st.warning("Não foi possível identificar as mensagens para avaliação.")
                else:
                    st.warning("Selecione uma nota válida entre 1 e 5.")
            else:
                st.info("Avaliação já enviada para esta interação.")
    with col4:
        if st.button("Iniciar uma nova conversa", key=f"clear_{model_key}"):
            keys_to_reset = [
                f"{model_key}_messages",
                f"{model_key}_last_input_tokens",
                f"{model_key}_last_output_tokens",
                f"{model_key}_last_time",
                f"{model_key}_ratings",
                f"{model_key}_rating_submitted"
            ]
            for key in keys_to_reset:
                if key in st.session_state:
                    del st.session_state[key]
            st.rerun()
    with col5:
        if st.button("Enviar para montagem do Recurso", key=f"send_for_resource_{model_key}"):
            messages = st.session_state.get(f"{model_key}_messages", [])
            last_assistant = next((msg["content"] for msg in reversed(messages) if msg["role"]=="assistant"), None)
            if last_assistant:
                st.session_state["texto_recurso"] = last_assistant
                st.success("Texto enviado para montagem do recurso!")
            else:
                st.warning("Nenhuma resposta do modelo disponível para enviar.")

# Interface de chat global para todos os modelos
def global_chat_interface():
    if "global_messages" not in st.session_state:
        st.session_state.global_messages = []
    with st.expander("Histórico de Conversa (Todos os modelos)", expanded=True):
        for msg in st.session_state.global_messages:
            st.markdown(
                f"<div style='background-color:#e6f7ff; padding:10px; margin-bottom:10px; border-radius:5px;'>"
                f"<strong>Você:</strong> {msg['content']}</div>",
                unsafe_allow_html=True
            )
    input_counter = st.session_state.get("global_input_counter", 0)
    user_input_key = f"global_input_{input_counter}"
    user_input = st.text_area(
        "Digite sua mensagem para todos os modelos:",
        key=user_input_key,
        height=180,
        placeholder="Escreva sua mensagem para enviar a todos os modelos..."
    )
    col1, col4 = st.columns([1.5, 2])
    with col1:
        if st.button("Enviar Mensagem para todos os modelos", key="send_all"):
            if user_input.strip():
                st.session_state.global_messages.append({"role": "user", "content": user_input.strip()})
                models_to_send = [
                    ('gpt', analyze_with_gpt),
                    ('gpt4', analyze_with_gpt4),
                    ('claude', analyze_with_claude),
                    ('maritalk', analyze_maritalk),
                    ('llama405b', analyze_llama405b),
                    ('llama70b_nebius', analyze_llama70b_nebius),
                    ('llama70b_groq', analyze_llama70b_groq),
                    ('deepseek_chat', analyze_deepseek_chat),
                    ('deepseek_reasoner', analyze_deepseek_reasoner),
                    ('gemini', analyze_with_gemini),
                    ('o3-mini-high', analyze_with_o3_mini_high),
                    ('qwen-max', analyze_with_qwen_max),
                    ('qwen-plus', analyze_with_qwen_plus)
                ]
                for model_key, _ in models_to_send:
                    messages_key = f"{model_key}_messages"
                    if messages_key not in st.session_state:
                        st.session_state[messages_key] = []
                    st.session_state[messages_key].append({"role": "user", "content": user_input.strip()})
                st.session_state.global_input_counter = input_counter + 1
                async def process_all_models():
                    tasks = []
                    for model_key, analysis_func in models_to_send:
                        messages = st.session_state[f"{model_key}_messages"]
                        tasks.append(analysis_func(messages))
                    return await asyncio.gather(*tasks)
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                try:
                    with st.spinner("Processando todos os modelos..."):
                        responses = loop.run_until_complete(process_all_models())
                finally:
                    loop.close()
                for (model_key, _), (response, input_tokens, output_tokens, elapsed_time) in zip(models_to_send, responses):
                    if response:
                        st.session_state[f"{model_key}_messages"].append({"role": "assistant", "content": response})
                        st.session_state[f"{model_key}_last_input_tokens"] = input_tokens
                        st.session_state[f"{model_key}_last_output_tokens"] = output_tokens
                        st.session_state[f"{model_key}_last_time"] = elapsed_time
                st.rerun()
    with col4:
        if st.button("Iniciar uma nova conversa", key="clear_global"):
            if 'global_messages' in st.session_state:
                del st.session_state.global_messages
            st.session_state.global_input_counter = 0
            st.rerun()

# Interface principal
def main_interface():
    st.title("Ambiente de Validação dos modelos de IA")
    if "resultados" not in st.session_state:
        st.session_state["resultados"] = []
    tabs = st.tabs([
        "Todos os modelos",
        "GPT-4o-Mini",
        "GPT-4",
        "Claude-3.5-Sonnet",
        "Maritalk Sabiá-3",
        "Llama 405B",
        "Llama 70B (Nebius)",
        "Llama 70B (Groq)",
        "Deepseek Chat",
        "Deepseek Reasoner",
        "Gemini 2.0 Flash",
        "O3-Mini-high",
        "Qwen-Max",
        "Qwen-Plus",
        "Resultados",
        "Gráficos",
        "Texto do Recurso"
    ])
    with tabs[0]:
        global_chat_interface()
    with tabs[1]:
        chat_interface('gpt', 'GPT-4o-Mini', analyze_with_gpt)
    with tabs[2]:
        chat_interface('gpt4', 'GPT-4', analyze_with_gpt4)
    with tabs[3]:
        chat_interface('claude', 'Claude-3.5-Sonnet', analyze_with_claude)
    with tabs[4]:
        chat_interface('maritalk', 'Maritalk Sabiá-3', analyze_maritalk)
    with tabs[5]:
        chat_interface('llama405b', 'Llama 405B', analyze_llama405b)
    with tabs[6]:
        chat_interface('llama70b_nebius', 'Llama 70B (Nebius)', analyze_llama70b_nebius)
    with tabs[7]:
        chat_interface('llama70b_groq', 'Llama 70B (Groq)', analyze_llama70b_groq)
    with tabs[8]:
        chat_interface('deepseek_chat', 'Deepseek Chat', analyze_deepseek_chat)
    with tabs[9]:
        chat_interface('deepseek_reasoner', 'Deepseek Reasoner', analyze_deepseek_reasoner)
    with tabs[10]:
        chat_interface('gemini', 'Gemini 2.0 Flash', analyze_with_gemini)
    with tabs[11]:
        chat_interface('o3-mini-high', 'O3-Mini-high', analyze_with_o3_mini_high)
    with tabs[12]:
        chat_interface('qwen-max', 'Qwen-Max', analyze_with_qwen_max)
    with tabs[13]:
        chat_interface('qwen-plus', 'Qwen-Plus', analyze_with_qwen_plus)
    with tabs[14]:
        st.subheader("Resultados das Avaliações")
        resultados = st.session_state.get("resultados", [])
        if resultados:
            df_resultados = pd.DataFrame(resultados)
            st.markdown("#### Sessão Atual")
            st.dataframe(df_resultados, use_container_width=True)
        else:
            st.info("Nenhuma avaliação enviada na sessão atual.")
        st.markdown("---")
        st.subheader("Upload de Resultados Históricos")
        uploaded_file = st.file_uploader("Carregue um arquivo CSV ou Excel com resultados anteriores", type=["csv", "xlsx"])
        if uploaded_file is not None:
            try:
                if uploaded_file.name.endswith('.csv'):
                    df_uploaded = pd.read_csv(uploaded_file, sep=';')
                else:
                    df_uploaded = pd.read_excel(uploaded_file)
                st.session_state["historical_results"] = df_uploaded.to_dict('records')
                st.success("Resultados históricos carregados com sucesso!")
            except Exception as e:
                st.error(f"Erro ao carregar arquivo: {e}")
        historical = st.session_state.get("historical_results", [])
        combined = historical + st.session_state.get("resultados", [])
        if combined:
            df_combined = pd.DataFrame(combined)
            st.markdown("#### Resultados Combinados")
            st.dataframe(df_combined, use_container_width=True)
            towrite = BytesIO()
            df_combined.to_excel(towrite, index=False, engine='openpyxl')
            towrite.seek(0)
            st.download_button(
                label="Baixar Relatório Combinado em .xlsx",
                data=towrite,
                file_name="resultados_combinados.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )
        else:
            st.info("Nenhum resultado para exibir (sessão atual e/ou histórico).")
    with tabs[15]:
        st.subheader("Gráficos das Avaliações (Resultados Combinados)")
        historical = st.session_state.get("historical_results", [])
        session_results = st.session_state.get("resultados", [])
        combined = historical + session_results
        if combined:
            df = pd.DataFrame(combined)
            df['Tempo_num'] = df['Tempo'].str.replace(' s', '').astype(float)
            df['Custo_num'] = df['Custo (R$)'].str.replace('R$', '', regex=False).astype(float)
            modelos_filtro = st.multiselect("Selecione os modelos:", options=df["modelo"].unique(), default=df["modelo"].unique())
            df_filtered = df[df["modelo"].isin(modelos_filtro)]
            
            st.markdown("### Avaliação Média vs Custo Médio por Modelo")
            df_summary = df_filtered.groupby('modelo').agg({
                'nota obtida': 'mean',
                'Custo_num': 'mean',
                'resultado': 'count'
            }).reset_index().rename(columns={'resultado': 'interações'})
            fig_scatter = px.scatter(
                df_summary,
                x='nota obtida',
                y='Custo_num',
                color='modelo',
                size='interações',
                hover_name='modelo',
                labels={'nota obtida': 'Avaliação Média', 'Custo_num': 'Custo Médio (R$)'},
                title="Avaliação Média vs Custo Médio por Modelo"
            )
            st.plotly_chart(fig_scatter, use_container_width=True)
            
            st.markdown("### Avaliação Média vs Tempo Médio de Resposta por Modelo")
            df_summary_time = df_filtered.groupby('modelo').agg({
                'nota obtida': 'mean',
                'Tempo_num': 'mean',
                'resultado': 'count'
            }).reset_index().rename(columns={'resultado': 'interações'})
            fig_time_scatter = px.scatter(
                df_summary_time,
                x='nota obtida',
                y='Tempo_num',
                color='modelo',
                size='interações',
                hover_name='modelo',
                labels={'nota obtida': 'Avaliação Média', 'Tempo_num': 'Tempo Médio (s)'},
                title="Avaliação Média vs Tempo Médio de Resposta por Modelo"
            )
            st.plotly_chart(fig_time_scatter, use_container_width=True)
            
            st.markdown("### Tempo Médio de Resposta por Modelo")
            tempo_media = df_filtered.groupby('modelo')['Tempo_num'].mean().reset_index()
            fig_tempo = px.bar(
                tempo_media,
                x='modelo',
                y='Tempo_num',
                labels={'modelo':'Modelo', 'Tempo_num':'Tempo Médio (s)'},
                title="Tempo Médio de Resposta"
            )
            st.plotly_chart(fig_tempo, use_container_width=True)
            
            st.markdown("### Custo Médio por Interação por Modelo")
            custo_media = df_filtered.groupby('modelo')['Custo_num'].mean().reset_index()
            fig_custo = px.bar(
                custo_media,
                x='modelo',
                y='Custo_num',
                labels={'modelo':'Modelo', 'Custo_num':'Custo Médio (R$)'},
                title="Custo Médio por Interação"
            )
            st.plotly_chart(fig_custo, use_container_width=True)
        else:
            st.info("Nenhuma avaliação para exibir gráficos.")
    with tabs[16]:
        st.subheader("Texto do Recurso")
        texto = st.session_state.get("texto_recurso", "")
        if texto:
            st.markdown(
                f"<div style='background-color:#f9f9f9; padding:15px; border: 1px solid #ccc; border-radius:5px;'>{texto}</div>",
                unsafe_allow_html=True
            )
        else:
            st.info("Nenhum texto enviado para montagem do recurso.")

# Execução principal
if __name__ == "__main__":
    if 'clients' not in st.session_state:
        st.session_state.clients = init_clients()
    main_interface()
