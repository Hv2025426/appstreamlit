# Modelo com Acionamento Assíncrono de Análises - várias APIs de IA simultaneas e relatório de uso de tokens

import os
import asyncio
import time
import pandas as pd
import tiktoken
from dotenv import load_dotenv
from openai import OpenAI
import anthropic
from groq import Groq
import google.generativeai as genai
import streamlit as st

# Configuração da página DEVE SER A PRIMEIRA CHAMADA
st.set_page_config(page_title="Analisador de Texto", layout="wide")

# Carregar variáveis de ambiente
load_dotenv()

# Inicialização de clientes
@st.cache_resource
def init_clients():
    """Inicializa clientes para diferentes serviços de IA"""
    genai.configure(api_key=os.getenv('GOOGLE_API_KEY'))
    
    return {
        'openai': OpenAI(api_key=os.getenv('OPENAI_API_KEY')),
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

# Funções de análise para cada modelo
async def analyze_with_gpt(prompt):
    start_time = time.time()
    try:
        response = await asyncio.to_thread(
            st.session_state.clients['openai'].chat.completions.create,
            model="gpt-4o-mini",
            temperature=0.5,
            messages=[
                {"role": "system", "content": "Você é um analista de textos especializado em língua portuguesa"},
                {"role": "user", "content": prompt}
            ]
        )
        st.session_state.gpt_result = response.choices[0].message.content
        st.session_state.gpt_input = response.usage.prompt_tokens
        st.session_state.gpt_output = response.usage.completion_tokens
        st.session_state.gpt_time = time.time() - start_time
    except Exception as e:
        st.error(f"Erro no GPT: {str(e)}")

async def analyze_with_claude(prompt):
    start_time = time.time()
    try:
        response = await asyncio.to_thread(
            st.session_state.clients['claude'].messages.create,
            model="claude-3-5-sonnet-20241022",
            max_tokens=2000,
            temperature=0.5,
            messages=[{"role": "user", "content": prompt}]
        )
        st.session_state.claude_result = response.content[0].text
        st.session_state.claude_input = response.usage.input_tokens
        st.session_state.claude_output = response.usage.output_tokens
        st.session_state.claude_time = time.time() - start_time
    except Exception as e:
        st.error(f"Erro no Claude: {str(e)}")

async def analyze_maritalk(prompt):
    start_time = time.time()
    try:
        response = await asyncio.to_thread(
            st.session_state.clients['maritalk'].chat.completions.create,
            model="sabia-3",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=8000,
            temperature=0.7
        )
        st.session_state.maritalk_result = response.choices[0].message.content
        encoding = tiktoken.get_encoding("cl100k_base")
        st.session_state.maritalk_input = len(encoding.encode(prompt))
        st.session_state.maritalk_output = len(encoding.encode(response.choices[0].message.content))
        st.session_state.maritalk_time = time.time() - start_time
    except Exception as e:
        st.error(f"Erro no Maritalk: {str(e)}")

async def analyze_llama405b(prompt):
    start_time = time.time()
    try:
        response = await asyncio.to_thread(
            st.session_state.clients['nebius_405b'].chat.completions.create,
            model="meta-llama/Meta-Llama-3.1-405B-Instruct",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=5000,
            temperature=0.6
        )
        st.session_state.llama405b_result = response.choices[0].message.content
        encoding = tiktoken.get_encoding("cl100k_base")
        st.session_state.llama405b_input = len(encoding.encode(prompt))
        st.session_state.llama405b_output = len(encoding.encode(response.choices[0].message.content))
        st.session_state.llama405b_time = time.time() - start_time
    except Exception as e:
        st.error(f"Erro no Llama 405B: {str(e)}")

async def analyze_llama70b_nebius(prompt):
    start_time = time.time()
    try:
        response = await asyncio.to_thread(
            st.session_state.clients['nebius_70b'].chat.completions.create,
            model="meta-llama/Llama-3.3-70B-Instruct",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=5000,
            temperature=0.6
        )
        st.session_state.llama70b_nebius_result = response.choices[0].message.content
        encoding = tiktoken.get_encoding("cl100k_base")
        st.session_state.llama70b_nebius_input = len(encoding.encode(prompt))
        st.session_state.llama70b_nebius_output = len(encoding.encode(response.choices[0].message.content))
        st.session_state.llama70b_nebius_time = time.time() - start_time
    except Exception as e:
        st.error(f"Erro no Llama 70B (Nebius): {str(e)}")

async def analyze_llama70b_groq(prompt):
    start_time = time.time()
    try:
        response = await asyncio.to_thread(
            st.session_state.clients['groq'].chat.completions.create,
            model="llama-3.3-70b-versatile",
            messages=[
                {"role": "system", "content": "Você é um analista de textos especializado em língua portuguesa"},
                {"role": "user", "content": prompt}
            ]
        )
        st.session_state.llama70b_groq_result = response.choices[0].message.content
        encoding = tiktoken.get_encoding("cl100k_base")
        st.session_state.llama70b_groq_input = len(encoding.encode(prompt))
        st.session_state.llama70b_groq_output = len(encoding.encode(response.choices[0].message.content))
        st.session_state.llama70b_groq_time = time.time() - start_time
    except Exception as e:
        st.error(f"Erro no Llama 70B (Groq): {str(e)}")

async def analyze_deepseek_chat(prompt):
    start_time = time.time()
    try:
        response = await asyncio.to_thread(
            st.session_state.clients['deepseek'].chat.completions.create,
            model="deepseek-chat",
            messages=[
                {"role": "system", "content": "Você é um analista de textos especializado em língua portuguesa"},
                {"role": "user", "content": prompt}
            ],
            max_tokens=8192,
            temperature=0.7
        )
        st.session_state.deepseek_chat_result = response.choices[0].message.content
        encoding = tiktoken.get_encoding("cl100k_base")
        st.session_state.deepseek_chat_input = len(encoding.encode(prompt))
        st.session_state.deepseek_chat_output = len(encoding.encode(response.choices[0].message.content))
        st.session_state.deepseek_chat_time = time.time() - start_time
    except Exception as e:
        st.error(f"Erro no Deepseek Chat: {str(e)}")

async def analyze_deepseek_reasoner(prompt):
    start_time = time.time()
    try:
        response = await asyncio.to_thread(
            st.session_state.clients['deepseek'].chat.completions.create,
            model="deepseek-reasoner",
            messages=[
                {"role": "system", "content": "Você é um analista de textos especializado em língua portuguesa"},
                {"role": "user", "content": prompt}
            ],
            max_tokens=8192,
            temperature=0.7
        )
        st.session_state.deepseek_reasoner_result = response.choices[0].message.content
        encoding = tiktoken.get_encoding("cl100k_base")
        st.session_state.deepseek_reasoner_input = len(encoding.encode(prompt))
        st.session_state.deepseek_reasoner_output = len(encoding.encode(response.choices[0].message.content))
        st.session_state.deepseek_reasoner_time = time.time() - start_time
    except Exception as e:
        st.error(f"Erro no Deepseek Reasoner: {str(e)}")

async def analyze_with_gemini(prompt):
    start_time = time.time()
    try:
        model = st.session_state.clients['gemini'].GenerativeModel("gemini-1.5-flash")
        response = await asyncio.to_thread(
            model.generate_content,
            prompt
        )
        st.session_state.gemini_result = response.text
        encoding = tiktoken.get_encoding("cl100k_base")
        st.session_state.gemini_input = len(encoding.encode(prompt))
        st.session_state.gemini_output = len(encoding.encode(response.text))
        st.session_state.gemini_time = time.time() - start_time
    except Exception as e:
        st.error(f"Erro no Gemini: {str(e)}")

# Funções auxiliares
def create_prompt(text):
    return f"""
    Analise o seguinte texto considerando estes elementos de coesão:
    - Retomada pronominal
    - Substituição lexical
    - Emprego de articuladores
    - Progressão textual

    Texto para análise:
    {text}
    """

def token_report():
    models = [
        ('GPT-4o-Mini', 'gpt_input', 'gpt_output', 'gpt_time'),
        ('Claude-3.5-Sonnet', 'claude_input', 'claude_output', 'claude_time'),
        ('Maritalk Sabiá-3', 'maritalk_input', 'maritalk_output', 'maritalk_time'),
        ('Llama 405B', 'llama405b_input', 'llama405b_output', 'llama405b_time'),
        ('Llama 70B (Nebius)', 'llama70b_nebius_input', 'llama70b_nebius_output', 'llama70b_nebius_time'),
        ('Llama 70B (Groq)', 'llama70b_groq_input', 'llama70b_groq_output', 'llama70b_groq_time'),
        ('Deepseek Chat', 'deepseek_chat_input', 'deepseek_chat_output', 'deepseek_chat_time'),
        ('Deepseek Reasoner', 'deepseek_reasoner_input', 'deepseek_reasoner_output', 'deepseek_reasoner_time'),
        ('Gemini 1.5 Flash', 'gemini_input', 'gemini_output', 'gemini_time')
    ]
    
    data = {
        'Modelo': [],
        'Input Tokens': [],
        'Output Tokens': [],
        'Total Tokens': [],
        'Tempo de Resposta': []
    }
    
    for model in models:
        input_tokens = st.session_state.get(model[1], 0)
        output_tokens = st.session_state.get(model[2], 0)
        response_time = st.session_state.get(model[3], 0.0)
        
        data['Modelo'].append(model[0])
        data['Input Tokens'].append(input_tokens)
        data['Output Tokens'].append(output_tokens)
        data['Total Tokens'].append(input_tokens + output_tokens)
        data['Tempo de Resposta'].append(response_time)
    
    return pd.DataFrame(data)

# Interface principal
def main_interface():
    tabs = [
        "Entrada", "GPT-4o-Mini", "Claude-3.5-Sonnet", 
        "Maritalk Sabiá-3", "Llama 405B", "Llama 70B (Nebius)",
        "Llama 70B (Groq)", "Deepseek Chat", "Deepseek Reasoner",
        "Gemini 1.5 Flash", "Estatísticas"
    ]
    tab_objects = st.tabs(tabs)
    
    with tab_objects[0]:
        user_input = st.text_area("Cole seu texto aqui:", height=300, key="text_input")
        if st.button("Analisar Texto"):
            if user_input.strip():
                st.session_state['analise_texto'] = create_prompt(user_input)
                run_analysis()
            else:
                st.warning("Por favor, insira um texto para análise.")
    
    if 'analise_texto' in st.session_state:
        def show_result(tab_index, model_key):
            with tab_objects[tab_index]:
                if model_key in st.session_state:
                    st.markdown(st.session_state[model_key])
                else:
                    st.info(f"Aguardando análise {model_key.split('_')[0]}...")
        
        show_result(1, 'gpt_result')
        show_result(2, 'claude_result')
        show_result(3, 'maritalk_result')
        show_result(4, 'llama405b_result')
        show_result(5, 'llama70b_nebius_result')
        show_result(6, 'llama70b_groq_result')
        show_result(7, 'deepseek_chat_result')
        show_result(8, 'deepseek_reasoner_result')
        show_result(9, 'gemini_result')
        
        with tab_objects[10]:
            st.subheader("Estatísticas de Uso")
            if any(key in st.session_state for key in ['gpt_input', 'claude_input']):
                df = token_report()
                st.dataframe(
                    df.style.format({
                        'Tempo de Resposta': '{:.1f}s'
                    }),
                    use_container_width=True,
                    column_config={
                        "Modelo": "Modelo",
                        "Input Tokens": st.column_config.NumberColumn("Tokens de Entrada"),
                        "Output Tokens": st.column_config.NumberColumn("Tokens de Saída"),
                        "Total Tokens": st.column_config.NumberColumn("Total de Tokens"),
                        "Tempo de Resposta": st.column_config.NumberColumn("Tempo de Resposta (s)")
                    }
                )
                st.caption("Detalhamento de desempenho por modelo")
            else:
                st.info("Execute uma análise para ver as estatísticas de uso")

# Função de execução
def run_analysis():
    keys_to_reset = [k for k in st.session_state.keys() if k.endswith(('result', 'input', 'output', 'time'))]
    for key in keys_to_reset:
        if key in st.session_state:
            del st.session_state[key]
    
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        with st.spinner("Processando análises simultâneas..."):
            loop.run_until_complete(asyncio.gather(
                analyze_with_gpt(st.session_state.analise_texto),
                analyze_with_claude(st.session_state.analise_texto),
                analyze_maritalk(st.session_state.analise_texto),
                analyze_llama405b(st.session_state.analise_texto),
                analyze_llama70b_nebius(st.session_state.analise_texto),
                analyze_llama70b_groq(st.session_state.analise_texto),
                analyze_deepseek_chat(st.session_state.analise_texto),
                analyze_deepseek_reasoner(st.session_state.analise_texto),
                analyze_with_gemini(st.session_state.analise_texto)
            ))
    finally:
        loop.close()
    st.rerun()

# Execução principal
if __name__ == "__main__":
    if 'clients' not in st.session_state:
        st.session_state.clients = init_clients()
    
    main_interface()