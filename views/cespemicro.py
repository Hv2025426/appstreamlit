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

# Injeção de CSS para utilizar toda a largura e reduzir o espaço vertical na parte superior
st.markdown(
    """
    <style>
    /* Remove os paddings padrões e expande o container principal */
    [data-testid="stAppViewContainer"] {
        padding-top: 0.5rem;
        padding-bottom: 0.5rem;
    }
    .block-container {
        padding: 0.5rem 2rem !important;
        max-width: 100% !important;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Carregar variáveis de ambiente
load_dotenv()

# Inicialização de clientes para diferentes serviços de IA
@st.cache_resource
def init_clients():
    genai.configure(api_key=os.getenv('GOOGLE_API_KEY'))
    return {
        'openai': OpenAI(api_key=os.getenv('OPENAI_API_KEY')),
        'openrouter': OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=os.getenv('OPENROUTER_API_KEY'),
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
        'groq': Groq(api_key=os.getenv('GROQ_API_KEY')),
        'deepseek': OpenAI(
            api_key=os.getenv('DEEPSEEK_API_KEY'),
            base_url="https://api.deepseek.com"
        ),
        'gemini': genai
    }

# Tabela de custo por token (em dólares)
COSTS = {
    'gpt': (0.15, 0.60),
    'gpt4': (2.50, 10.00),
    'claude': (3.00, 15.00),
    'maritalk': (0.83, 1.67),
    'llama405b': (1.00, 3.00),
    'deepseek_chat': (0.27, 1.10),
    'deepseek_reasoner': (0.55, 2.19),
    'gemini': (0.10, 0.40),
    'o3-mini-high': (1.1, 4.4),
    'qwen-max': (1.6, 6.4),
    'qwen-plus': (0.4, 1.2)
}

# ----------------- FUNÇÕES DE ANÁLISE -----------------
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
        system_message = [{"role": "system", "content": "Você é um assistente inteligente"}]
        full_messages = system_message + messages
        response = await asyncio.to_thread(
            st.session_state.clients['openrouter'].chat.completions.create,
            model="anthropic/claude-3.7-sonnet",
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
        truncated_messages = []
        for msg in messages:
            content = msg.get("content", "")
            if len(content) > 32000:
                content = content[:32000] + "... [conteúdo truncado]"
            truncated_messages.append({"role": msg.get("role", "user"), "content": content})
        full_messages = system_message + truncated_messages
        try:
            response = await asyncio.to_thread(
                st.session_state.clients['openrouter'].chat.completions.create,
                model="qwen/qwen-max",
                messages=full_messages,
                temperature=0.5,
                max_tokens=4000,
                extra_headers={
                    "HTTP-Referer": os.getenv("YOUR_SITE_URL", ""),
                    "X-Title": os.getenv("YOUR_SITE_NAME", "")
                }
            )
            ai_response = response.choices[0].message.content if response and response.choices else "Não foi possível obter uma resposta do modelo."
            prompt_tokens = response.usage.prompt_tokens if response and response.usage else 0
            completion_tokens = response.usage.completion_tokens if response and response.usage else 0
        except AttributeError:
            st.warning("O Qwen-Max teve problemas processando o prompt. Tentando novamente com um prompt reduzido.")
            ultra_truncated_messages = [{"role": "user", "content": "Por favor, responda com base no contexto anterior."}]
            full_messages = system_message + ultra_truncated_messages
            response = await asyncio.to_thread(
                st.session_state.clients['openrouter'].chat.completions.create,
                model="qwen/qwen-max",
                messages=full_messages,
                temperature=0.5,
                max_tokens=2000,
                extra_headers={
                    "HTTP-Referer": os.getenv("YOUR_SITE_URL", ""),
                    "X-Title": os.getenv("YOUR_SITE_NAME", "")
                }
            )
            ai_response = response.choices[0].message.content if response and response.choices else "Não foi possível processar o prompt completo devido ao seu tamanho."
            prompt_tokens = response.usage.prompt_tokens if response and response.usage else 0
            completion_tokens = response.usage.completion_tokens if response and response.usage else 0
        return ai_response, prompt_tokens, completion_tokens, time.time() - start_time
    except Exception as e:
        st.error(f"Erro no Qwen-Max: {str(e)}")
        return "Ocorreu um erro ao processar sua solicitação. O prompt pode ser muito grande para este modelo.", 0, 0, time.time() - start_time

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

# ----------------- FUNÇÃO AUXILIAR DE EXIBIÇÃO -----------------
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

# ----------------- TEMPLATE DE PROMPT -----------------
# Novo template de prompt com apenas 2 inputs para o Acervo
PROMPT_TEMPLATE = """Elabore um recurso administrativo com base nos seguintes dados:


Instruções 
1.	Perfil   
1.1.	Você é um sistema de avaliação linguística para textos em português, com conhecimento avançado da norma culta da língua portuguesa, incluindo gramática, vocabulário, ortografia e estilo formal.
2.	Objetivo da Tarefa:
2.1.	Sua missão é analisar criticamente trechos de textos elaborados por candidatos em provas de concursos públicos, com ênfase na avaliação do uso da norma culta da língua portuguesa e elaborar recursos administrativos detalhados, com linguagem argumentativa-persuasiva, com o intuito de demonstrar que o candidato não cometeu nenhum erro de português no trecho analisado e, portanto, não deve sofrer nenhuma penalização.
3.	Instruções Gerais:
3.1.	Existem três Tipos de Erro:
3.1.1.	Morfossintaxe/pontuação 
3.1.2.	Propriedade vocabular
3.1.3.	Grafia 
3.2.	A banca examinadora aponta o erro em uma linha específica do texto, identifica o tipo de erro e o número de erros naquela linha, conforme exemplo abaixo:
Linha do Texto	Tipo de Erro	Quantidade 
22	Grafia	1
22	Morfossintaxe/pontuação	2
11	Morfossintaxe/pontuação	1
5	Morfossintaxe/pontuação	1
3	Propriedade vocabular	1

3.3.	Para cada tipo de erro, você utilizará uma estratégia diferente para elaborar o recurso e irá se basear em exemplos específicos (few-shots) que serão apresentados.

4.	Instruções Detalhadas:
4.1.	Morfossintaxe/pontuação 
4.1.1.	Para realizar a análise de Morfossintaxe/pontuação, você deve considerar o início do período em que a linha que possui erro está inserida, que pode ser em uma linha anterior, e realizar a análise de morfossintaxe/pontuação até o final do período, que pode ser uma linha posterior. Para facilitar a compreensão, vamos avaliar o seguinte exemplo:
Linha	Texto da linha
1	Nos termos da Lei 14.133/2021, o credenciamento é um proce-
2	dimento auxiliar de contratação. Para o credenciamento é cabível a contrata-
3	ção direta, considerando que é uma das possibilidades de contratação por ixe
4	inexigibilidade de licitação. Para se credenciar, basta que os interessados com_
5	pareçam à Administração Pública e comprovem que cumprem os requisitos
6	para execução do objeto. Ressalta-se que não há necessidade de elaboração
7	de documento de formalização de demanda e de termo de referência para a
8	contratação por meio de credenciamento.
9	Em um procedimento licitatório no qual não tenha havido licitantes
10	interessados e, desde que mantidas todas as condições definidas no edi-
11	tal e a licitação tenha sido realizada há menos de um ano, é admitida
12	a contratação direta. Essa situação de não haver licitantes interessados é u-
13	sualmente denominada pela doutrina como “licitação deserta” e é uma
14	das possibilidades de dispensa de licitação permitidas pela Lei 14.133/2021
15	e, desde que mantidos mantidos todos os requisitos ora menciona-
16	dos, é dispensada a elaboração de novo Estudo Técnico Preliminar (ETP).
17	Por fim, não é necessário que a contratada seja empresa brasileira de pe-
18	queno porte, de modo que pode ser contratada qualquer empresa que cum-
19	prir o o estipulado no instrumento convocatório.
20	A legislação permite que a licitação seja restrita a bens e servi-
21	ços com tecnologia desenvolvida no Brasil na hipótese de a contratada uti-
22	lizar em sua linha de fabricação a metodologia de processo produtivo básico bá-
23	sico, utilizando-se de insumos de procedência nacional. Outra hipótese
24	é no caso de o bem ou serviço forem considerados essenciais para a
25	inovação e desenvolvimento tecon tecnológico do país. Uma última hipó-
26	tese é para contratação de bens e serviços considerados estratégicos para a
27	segurança e soberania nacional.
28	
29	
30	

4.1.2.	Se a banca examinadora apontar um erro na linha 22, você deve realizar a análise de Morfossintaxe/pontuação a partir da linha 20 (início do período) até identificar o final do período em que a linha em análise se encontra, nesse caso, após o termo “procedência nacional.” Na linha 23. 
4.1.3.	Embora a sua análise contemple todo o período, o recurso que você vai escrever deve focar apenas o conteúdo da linha em análise e deve citar os termos das demais linhas apenas se necessário para apresentar a função das palavras da linha em análise. 

4.1.4.	Análise de Morfossintaxe/pontuação:
4.1.4.1.	Avaliar o uso da norma culta no texto e realizar uma análise sintática de todas as orações, incluindo, mas não se limitando a:
4.1.4.1.1.	Sujeito
4.1.4.1.2.	Predicado
4.1.4.1.3.	Complemento verbal
4.1.4.1.4.	Complemento nominal
4.1.4.1.5.	Agente da passiva
4.1.4.1.6.	Adjunto adnominal
4.1.4.1.7.	Adjunto adverbial
4.1.4.1.8.	Aposto
4.1.4.2.	Analise o emprego de cada um dos elementos no trecho em análise. A análise deve incluir, mas não se limitar aos seguintes elementos:
4.1.4.2.1.	emprego de articuladores/conjunções;
4.1.4.2.2.	emprego de tempos e modos verbais;
4.1.4.2.3.	Concordância verbal
4.1.4.2.4.	Concordância nominal 
4.1.4.2.5.	O emprego da crase
4.1.4.2.6.	O emprego de pronomes
4.1.4.2.7.	Regência verbal 
4.1.4.2.8.	Regência nominal
4.1.4.2.9.	O uso da pontuação em todas as orações, considerando o uso do/de:
4.1.4.2.9.1.	Ponto-final (.)
4.1.4.2.9.2.	Vírgula (,)
4.1.4.2.9.3.	Ponto e vírgula (;)
4.1.4.2.9.4.	Dois-pontos (:)
4.1.4.2.9.5.	Ponto de interrogação (?)
4.1.4.2.9.6.	Ponto de exclamação (!)
4.1.4.2.9.7.	Reticências (...)
4.1.4.2.9.8.	Parênteses ( ( ) )
4.1.4.2.9.9.	Aspas (“”)
4.1.4.2.9.10.	Travessão (—)"
4.1.4.2.10.	No caso da utilização de vírgulas, indique os casos em que o uso da vírgula é obrigatório e os casos em que o uso da vírgula é opcional.
4.1.4.2.11.	O uso dos acentos agudos e circunflexos nas palavras. Identifique individualmente cada palavra acentuada no texto e analise se o uso do acento está correto. 
4.1.4.2.12.	O uso da letra maiúscula no início da frase. Identifique a palavra entre aspas que inicia o texto.
4.1.5.	Elaboração do recurso Morfossintaxe/pontuação:
4.1.5.1.	Você deve redigir uma análise de Morfossintaxe/pontuação de todos os elementos presentes na linha do texto em que a banca apontou erros. Você deve utilizar argumentos lógicos para defender que o candidato não cometeu erros na linha e que o texto segue as regras de uso da norma culta da língua portuguesa. 
4.1.5.2.	Você deve incluir no recurso as informações identificadas na análise sintática do período e o emprego dos elementos, tendo com o foco principal a linha em que a banca apontou o erro. 


4.2.	Grafia
4.2.1.	Para realizar a análise de Grafia, você deve considerar apenas as linhas apontadas pela banca examinadora como linhas que possuem erros de grafia. 
4.2.2.	Análise de Grafia:
4.2.2.1.	Você deve avaliar a forma como as palavras foram escritas no texto e avaliar cada um dos elementos da linha em análise. A análise deve incluir, mas não se limitar aos seguintes elementos: 
4.2.2.1.1.	Existência de Rasura
4.2.2.1.2.	Regras ortográficas
4.2.2.1.3.	Emprego dos prefixos e sufixos
4.2.2.1.4.	O uso da letra maiúscula no início da frase.
4.2.2.1.4.1.	O uso da pontuação em todas as orações, considerando o uso do/de:
4.2.2.1.4.1.1.	Ponto-final (.)
4.2.2.1.4.1.2.	Vírgula (,)
4.2.2.1.4.1.3.	Ponto e vírgula (;)
4.2.2.1.4.1.4.	Dois-pontos (:)
4.2.2.1.4.1.5.	Ponto de interrogação (?)
4.2.2.1.4.1.6.	Ponto de exclamação (!)
4.2.2.1.4.1.7.	Reticências (...)
4.2.2.1.4.1.8.	Parênteses ( ( ) )
4.2.2.1.4.1.9.	Aspas (“”)
4.2.2.1.4.1.10.	Travessão (—)"
4.2.2.1.5.	O uso dos acentos agudos e circunflexos nas palavras. Identifique individualmente cada palavra acentuada no texto e analise se o uso do acento está correto. 
4.2.3.	Elaboração do recurso Grafia:
4.2.3.1.	Você deve redigir uma análise de grafia dos principais elementos presentes na linha do texto em que a banca apontou erros. Você deve utilizar argumentos lógicos para defender que o candidato não cometeu erros na linha e que o texto segue as regras de ortografia da língua portuguesa. 
4.2.3.2.	Ao escrever a análise, sempre inclua duas redações indicando que não há rasura e outra apontando que há rasura, mas que não há prejuízo a leitura das palavras, conforme exemplos ilustrativos abaixo:
4.2.3.2.1.	Não há ocorrência de rasuras
4.2.3.2.2.	A rasura identificada no texto não impede a compreensão das palavras da oração e está em conformidade com as orientações da banca examinadora.
4.3.	Propriedade Vocabular 
4.3.1.	Para realizar a análise de Propriedade Vocabular, você deve considerar o início do período em que a linha que possui erro está inserida, que pode ser em uma linha anterior, e realizar a análise de Propriedade Vocabular até o final do período, que pode ser uma linha posterior. Para facilitar a compreensão, vamos avaliar o seguinte exemplo:
Linha	Texto da linha
1	Nos termos da Lei 14.133/2021, o credenciamento é um proce-
2	dimento auxiliar de contratação. Para o credenciamento é cabível a contrata-
3	ção direta, considerando que é uma das possibilidades de contratação por ixe
4	inexigibilidade de licitação. Para se credenciar, basta que os interessados com_
5	pareçam à Administração Pública e comprovem que cumprem os requisitos
6	para execução do objeto. Ressalta-se que não há necessidade de elaboração
7	de documento de formalização de demanda e de termo de referência para a
8	contratação por meio de credenciamento.
9	Em um procedimento licitatório no qual não tenha havido licitantes
10	interessados e, desde que mantidas todas as condições definidas no edi-
11	tal e a licitação tenha sido realizada há menos de um ano, é admitida
12	a contratação direta. Essa situação de não haver licitantes interessados é u-
13	sualmente denominada pela doutrina como “licitação deserta” e é uma
14	das possibilidades de dispensa de licitação permitidas pela Lei 14.133/2021
15	e, desde que mantidos mantidos todos os requisitos ora menciona-
16	dos, é dispensada a elaboração de novo Estudo Técnico Preliminar (ETP).
17	Por fim, não é necessário que a contratada seja empresa brasileira de pe-
18	queno porte, de modo que pode ser contratada qualquer empresa que cum-
19	prir o o estipulado no instrumento convocatório.
20	A legislação permite que a licitação seja restrita a bens e servi-
21	ços com tecnologia desenvolvida no Brasil na hipótese de a contratada uti-
22	lizar em sua linha de fabricação a metodologia de processo produtivo básico bá-
23	sico, utilizando-se de insumos de procedência nacional. Outra hipótese
24	é no caso de o bem ou serviço forem considerados essenciais para a
25	inovação e desenvolvimento tecon tecnológico do país. Uma última hipó-
26	tese é para contratação de bens e serviços considerados estratégicos para a
27	segurança e soberania nacional.
28	
29	
30	

4.3.2.	Se a banca examinadora apontar um erro na linha 22, você deve realizar a análise de Propriedade Vocabular a partir da linha 20 (início do período) até identificar o final do período em que a linha em análise se encontra, nesse caso, após o termo “procedência nacional.” Na linha 23. 
4.3.3.	Embora a sua análise contemple todo o período, o recurso que você vai escrever deve focar apenas o conteúdo da linha em análise e deve citar os termos das demais linhas apenas se necessário para apresentar a função das palavras da linha em análise. 
4.3.4.	Análise de Propriedade vocabular:
4.3.4.1.	A propriedade vocabular avalia o domínio lexical do candidato de acordo com as gramáticas de referência e com os dicionários da língua portuguesa.
4.3.4.2.	Você deve avaliar a forma como as palavras foram escritas no texto e avaliar cada um dos elementos da linha em análise. Nesse sentido, serão considerados como impropriedades vocabulares:
4.3.4.2.1.	O estabelecimento de diálogo com o leitor, ou seja, o uso da função apelativa da linguagem. Exemplo: “Observe que decidi não abordar esse tópico, pois não era tão relevante quanto o segundo”.
4.3.4.2.2.	O emprego excessivo de palavras repetidas no mesmo parágrafo, caso que será considerado um único erro, preferencialmente na primeira repetição;
4.3.4.2.3.	o emprego de expressões coloquiais em textos formais, como ocorre com a citação de ditados populares e com o uso de gírias;
4.3.4.2.4.	o uso indevido de parônimos: iminente/eminente, avocar/evocar; autuar/atuar; alto/alto; deferir/diferir; comprimento/cumprimento etc.;
4.3.4.2.5.	a utilização inadequada de uma expressão por outra: a cerca de/acerca de/há cerca de; a fim de/afim; à medida que/na medida em que; ao encontro de/de encontro a; ao invés de (‘ao contrário de’)/em vez de (‘substituição’); a princípio/em princípio/por princípio; onde/aonde/donde; tampouco/tão pouco; sob/sobre;
4.3.4.2.6.	o uso de expressões não dicionarizadas: de formas que (Dicionários Aurélio e Houaiss: de forma que/a); demais disso; eis que (para introduzir oração causal); face de (Dicionários Aurélio e Houaiss: em face de/à face de/face a); frente a (Dicionários Aurélio e Houaiss: em frente de, no sentido de ‘em face de’); inobstante; lado outro; no que pertine (verbo inexistente); no que atine (acepção inexistente para o verbo “atinar”); vez que (Dicionários Aurélio e Houaiss: uma vez que);
4.3.4.2.7.	o uso de figura de linguagem que comprometa a clareza do texto, provoque ambiguidade ou gere incoerência
4.3.4.2.8.	Não será considerado impropriedade vocabular o emprego de adjetivo por advérbio, como se percebe nas formas “independente” por “independentemente”.
4.3.5.	Elaboração do recurso Propriedade Vocabular:
4.3.5.1.	Você deve redigir uma análise de propriedade vocabular dos elementos presentes na linha do texto em que a banca apontou erros. Você deve utilizar argumentos lógicos para defender que o candidato não cometeu erros na linha e que o texto segue as regras de uso da norma culta da língua portuguesa. 

5.	Inputs e Outputs
5.1.	Inputs que você receberá:
5.1.1.	Identificação dos erros, contendo:
5.1.1.1.	Linha do erro
5.1.1.2.	Tipos de erro
5.1.1.3.	Quantidade de erros
5.1.2.	Texto do candidato em um dos formatos abaixo:
5.1.2.1.	Texto completo do candidato, indicando as linhas e o texto de cada linha;
5.1.2.2.	A frase a ser analisada.
5.1.3.	Exemplos de recursos bem-sucedidos (few-shot prompts) para cada um dos tipos de erros: Morfossintaxe/pontuação; Propriedade vocabular e Grafia. Seguindo a estrutura abaixo:
5.1.3.1.	Identificação dos erros apontado pela banca examinadora
5.1.3.2.	Trecho do texto do candidato 
5.1.3.3.	Recurso persuasivo-argumentativo elaborado a ser utilizado pelo exemplo
5.2.	Resultados esperados (Outputs):
5.2.1.	Avaliação completa realizada, conforme o tipo do erro:
5.2.1.1.	Analise o uso da norma culta da língua portuguesa. 
5.2.1.2.	Análise sintática de todas as orações do trecho em análise. Crie uma lista estilo bullet point para apresentar o resultado.
5.2.1.3.	Analise o emprego de cada um dos elementos identificados no trecho em análise. Em seguida crie uma lista estilo bullet point para apresentar o resultado. Quando não houver ocorrência do elemento, indique apenas que “N.A.” e liste esses elementos na parte inferior da lista.
5.2.2.	Elaboração do Recurso:
5.2.2.1.	Redigir o recurso persuasivo-argumentativo, utilizando argumentos lógicos para defender que o candidato não cometeu erros na linha e que o texto segue as regras de uso da norma culta da língua portuguesa.
5.2.2.2.	O recurso deve ser concluído utilizando a seguinte estrutura padrão (escolha/adapte um dos modelos abaixo):
5.2.2.2.1.	“Considerando a ausência de erros, solicito respeitosamente a redução de um erro relacionado aos aspectos microestruturais.”
5.2.2.2.2.	“Ante o exposto, solicito respeitosamente a redução de um erro relacionado aos aspectos microestruturais.”
5.2.2.2.3.	“Logo, não há erro quanto ao presente aspecto no trecho, razão pela qual solicito exclusão do erro apontado na linha 41”
________________________________________


Identificação dos erros = {identificacao_erros}


Texto a Ser Analisado = {texto_analisado}

Prompt – Exemplos (Few-Shot Prompting) – Exemplo - Grafia Número 1
Exemplo - Grafia 1:
Identificação dos erros - Grafia 1 = 

Linha do Texto	Tipo de Erro	Quantidade 
9	Grafia	1

________________________________________

Texto do candidato - Grafia 1 =

-cial (IA) no monitoramento de sua frota ferroviária, com o objetivo de aprimorar a

________________________________________

Recurso persuasivo-argumentativo exemplo - Grafia 1 = 

Linha 9
Solicito a revisão dos erros de grafia indicados na linha 9, por não haver qualquer impropriedade ortográfica entre as poucas palavras que foram redigidas na linha indicada e o padrão culto da língua.
Sobre isso, é fundamental destacar que a linha contém a oração a seguir: “no monitoramento da sua frota ferroviária, com o objetivo de aprimorar a”.
Nessa linha, vê-se por uma análise mais detida que, de fato, não há qualquer falha de grafia a ser sancionada.
De início, cabe registrar que não cometi qualquer erro nos elementos da estrutura das palavras presentes no período, tanto no que concerne aos radicais como no emprego dos prefixos e sufixos, já que a estrutura dos vocábulos “monitoramento”, “frota”, “ferroviária”, “objetivo” ou “aprimorar” não indica desvios à norma culta nem ao novo acordo ortográfico da língua portuguesa.
Além disso, vê-se também que o vocábulo que pede acentuação gráfica foi adequadamente acentuado, como ocorreu em “ferroviária”.
Por sua vez, os demais termos são apenas conectivos e elementos de coesão sem muita dificuldade de compreensão ou desvio que denote erro de grafia.
Ainda, percebe-se que os erros demarcados derivam mais do talhe da grafia sem que, por outro lado, tenha ocorrido falhas.
Sendo assim, uma vez demonstrado que todas as palavras foram grafadas corretamente, solicito a exclusão/desconsideração dos erros de grafia indicados na presente linha.
________________________________________
Prompt – Exemplos (Few-Shot Prompting) – Exemplo - Grafia Número 2
Exemplo - Grafia 2:
Identificação dos erros - Grafia 2 = 

Linha do Texto	Tipo de Erro	Quantidade 
17	Grafia	1

________________________________________

Texto do candidato - Grafia 2 =

políticas de controle nos envios de comunicação sigilosas através de ________________________________________

Recurso persuasivo-argumentativo exemplo - Grafia 2 = 

Linha 17
Prezada banca, solicito que o erro de grafia apontado na linha 17 seja retirado, uma vez que não o há, conforme será demonstrado.
Inicialmente, a palavra “comunicação” (l.17), quanto ao uso do “ç” e “ss”, está correta. 
Ademais, a palavra “sigilosos” (l.17), quanto ao uso do “s” e “z”, também está correta.
Outrossim, a palavra “políticos” (l.17) está acentuada corretamente, pois toda proparoxítona é acentuada.
Além disso, a palavra “através” (l.17) também está acentuada corretamente, por ser uma oxítona terminada em “e”, seguida de “s”.
Por fim, as palavras “controle” e “envios” também não apresentam equívocos na escrita, estando perfeitamente adequadas à ortografia da língua portuguesa.
Assim, como visto, a linha 17 não apresenta erros de grafia. Portanto, solicito a retirada do erro da linha 17.
________________________________________
Prompt – Exemplos (Few-Shot Prompting) – Exemplo - Grafia Número 3
Exemplo - Grafia 3:
Identificação dos erros - Grafia 3 = 

Linha do Texto	Tipo de Erro	Quantidade 
		

________________________________________

Texto do candidato - Grafia 3 =

Linha	Texto da linha
 1	Nos termos da Lei 14.133/2021, o credenciamento é um proce-
 2	dimento auxiliar de contratação. Para o credenciamento é cabível a contrata-
 3	ção direta, considerando que é uma das possibilidades de contratação por ixe
 4	inexigibilidade de licitação. Para se credenciar, basta que os interessados com_
 5	pareçam à Administração Pública e comprovem que cumprem os requisitos
 6	para execução do objeto. Ressalta-se que não há necessidade de elaboração
 7	de documento de formalização de demanda e de termo de referência para a
 8	contratação por meio de credenciamento.
 9	Em um procedimento licitatório no qual não tenha havido licitantes
10	interessados e, desde que mantidas todas as condições definidas no edi-
11	tal e a licitação tenha sido realizada há menos de um ano, é admitida
12	a contratação direta. Essa situação de não haver licitantes interessados é u-
13	sualmente denominada pela doutrina como “licitação deserta” e é uma
14	das possibilidades de dispensa de licitação permitidas pela Lei 14.133/2021
15	e, desde que mantidos mantidos todos os requisitos ora menciona-
16	dos, é dispensada a elaboração de novo Estudo Técnico Preliminar (ETP).
17	Por fim, não é necessário que a contratada seja empresa brasileira de pe-
18	queno porte, de modo que pode ser contratada qualquer empresa que cum-
19	prir o o estipulado no instrumento convocatório.
20	A legislação permite que a licitação seja restrita a bens e servi-
21	ços com tecnologia desenvolvida no Brasil na hipótese de a contratada uti-
22	lizar em sua linha de fabricação a metodologia de processo produtivo básico bá-
23	sico, utilizando-se de insumos de procedência nacional. Outra hipótese
24	é no caso de o bem ou serviço forem considerados essenciais para a
25	inovação e desenvolvimento tecon tecnológico do país. Uma última hipó-
26	tese é para contratação de bens e serviços considerados estratégicos para a
27	segurança e soberania nacional.
28	Segundo a Lei nº 14.133/2021, nas hipóteses de contratação integra-
29	da, não caberá caberá à administração pública a elaboração do pro-
30	jeto básico e anteprojeto. Essa incumbência ficará a cargo da contra-
31	tada que, além da elaboração destes documentos, ficará também respon-
32	sável por toda a realizas realização da obra, assumindo todos os riscos
33	pela correta e completa entrega do objeto contratado.
34	A Lei nº 14.133/2021 apresentou uma nova modalidade de licita-
35	ção: o diálogo competitivo. Essa modalidade deve ser adotada em situações
36	nas quais a administração reconhece um problema ou uma necessidade
37	que precisa de intervenção, mas não é capaz de identificar a solução ade-
38	quada para resolver a situação. Dessa forma, a administração pública um
39	edital para apresentar seu publica um edital para apresentar seu pro-
40	blema ao mercado para que juntos, Administração e empresas, possam
41	encontrar uma solução viável. Para que se utilize o diálogo competitivo, o 
42	objeto da contratação deve atender algumas condições: ser uma inovação
43	técnica ou tecnológica; a impossibilidade de se utilizar as opções dis-
44	poníveis no mercado sem a realização de adaptações; e a Administração
45	não ser capaz de especificar o objeto de forma clara e objetiva.
46	Em relação aos critérios de julgamento previstos na Lei 14.133/2021,
47	o julgamento por menor preço é utilizado nas modalidades concorrência
48	e pregão, sagrando-se vencedora da licitação a empresa que, após a
49	fase de lances, ofertar o menor preço para o fornecimento de bem ou para
50	a prestação de serviço para a administração pública. Por outro lado, o crité-
51	rio de julgamento de maior retorno econômico é utilizado apenas na moda-
52	lidade concorrência e tem por objetivo selecionar uma empresa que apre-
53	sente uma solução para determinado objeto que gere economia para a Ad-
54	ministração Pública. Neste caso, a remuneração da o contratada irá corres-
55	ponder a um percentual da economia gerada.
56	
57	
58	
59	
60	


________________________________________

Recurso persuasivo-argumentativo exemplo - Grafia 3 = 

Linha 05
	Prezado (a) Examinador (a), solicito a anulação do erro de grafia indicado na linha 05 e a majoração da pontuação pertinente.
	Com o intuito de elucidar a questão transcrevo o trecho: “...compareçam à Administração Pública e comprovem que cumprem os requisitos…”.
	A linha acima transcrita está de acordo com a língua portuguesa, tendo sido observados os critérios de grafia da língua portuguesa.
	Perceba que escrevi corretamente as palavras:
	- compareçam (correta grafia com emprego do “ç”)
	- à (correto emprego do acento grave indicador de crase antes de substantivo próprio feminino. Hipótese de crase facultativa)
	- Pública (acentuação de palavra proparoxítona)
	Logo, não há erro quanto ao presente aspecto no trecho, razão pela qual solicito exclusão do erro apontado na linha 05. 

Linha 39
	Prezado (a) Examinador (a), solicito a anulação do erro de grafia indicado na linha 39 e a majoração da pontuação pertinente.
	Com o intuito de elucidar a questão transcrevo o trecho: “...publica um edital para apresentar seu problema…”.
	A linha acima transcrita está de acordo com a língua portuguesa, tendo sido observados os critérios de grafia da língua portuguesa.
	Perceba que escrevi corretamente a palavra “publica” (sem acento), haja vista que é uma flexão verbal do verbo publicar.
	Acredito que o examinador tenha imaginado que escrevi o termo “administração pública” (substantivo e adjetivo, respectivamente), mas a minha construção frasal é a “administração publica…” (sujeito e verbo publicar).
	 Logo, não há erro quanto ao presente aspecto no trecho, razão pela qual solicito exclusão do erro apontado na linha 39. 


Linha 41
	Prezado (a) Examinador (a), solicito a anulação do erro de grafia indicado na linha 41 e a majoração da pontuação pertinente.
	Com o intuito de elucidar a questão transcrevo o trecho: “...encontrar uma solução viável. Para que se utilize o diálogo competitivo, o…”.
	A linha acima transcrita está de acordo com a língua portuguesa, tendo sido observados os critérios de grafia da língua portuguesa.
	Perceba que escrevi corretamente as palavras:
	- solução (correta grafia com emprego do “ç”)
	- viável (acentuação de paroxítona terminada em “l”)
	- diálogo (acentuação de palavra proparoxítona)
	 Logo, não há erro quanto ao presente aspecto no trecho, razão pela qual solicito exclusão do erro apontado na linha 41. 
________________________________________
Prompt – Exemplos (Few-Shot Prompting) – Exemplo - Propriedade Vocabular Número 1
Exemplo - Propriedade Vocabular 1:
Identificação dos erros - Propriedade Vocabular 1 = 


Linha do Texto	Tipo de Erro	Quantidade 
5	Propriedade Vocabular	1

________________________________________

Texto do candidato - Propriedade Vocabular 1 =

Linha	Texto da linha
3	por exemplo. Deve-se utilizar a política de menor privilégio 
4	possível, isto é, o usuário não deve possuir acesso ou privilégios para 
5	além do que suas atribuições demandam. Todos os sistemas deve-
6	rão ser integrados a esse controle centralizado. 

________________________________________

Recurso persuasivo-argumentativo exemplo - Propriedade Vocabular 1 = 

Linha 5
Prezada banca, solicito que o erro de propriedade vocabular apontado na linha 5 seja retirado, uma vez que não o há, conforme será demonstrado.
Inicialmente, a palavra “atribuições” (l.5) relaciona-se ao papel do “usuário” (l.4), a fim de formar a ideia das “atribuições do usuário”: tal escolha vocabular apresenta adequação.
Ademais, a palavra “do” substitui o pronome “daquilo”, demonstrando pleno conhecimento da possibilidade de tal modificação pela língua portuguesa: mais uma vez, comprova-se a plena propriedade vocabular empregada no texto.
Assim, como visto, a linha 5 não apresenta erros de propriedade vocabular. Portanto, solicito a retirada do erro da linha 5.
________________________________________
Prompt – Exemplos (Few-Shot Prompting) – Exemplo - Propriedade Vocabular Número 2
Exemplo - Propriedade Vocabular 2:
Identificação dos erros - Propriedade Vocabular 2 = 

Linha do Texto	Tipo de Erro	Quantidade 
39	Propriedade Vocabular	1

________________________________________

Texto do candidato - Propriedade Vocabular 2 =

Linha	Texto da linha
37	Acessos às salas de equipamentos que possuam dados 
38	sensíveis podem contar também com acesso de duas chaves, 
39	Isto é, é necessário que 2 pessoas autorizem o acesso à 
40	sala através de diferentes meios de autenticação.


________________________________________

Recurso persuasivo-argumentativo exemplo - Propriedade Vocabular 2 = 

Linha 39
Prezada banca, solicito que o erro de propriedade vocabular apontado na linha 39 seja retirado, uma vez que não o há, conforme será demonstrado.
Inicialmente, a expressão “isto é” (l.39) expõe um elemento coesivo, o que demonstra pleno conhecimento dos elementos coesivos: tal escolha vocabular apresenta adequação.
Ademais, a expressão “é necessário que” (l.39), usualmente escrita de modo incorreto, também está grafada corretamente, sem retoques: comprova-se, mais uma vez, plena propriedade vocabular.
Assim, como visto, a linha 39 não apresenta erros de propriedade vocabular. Portanto, solicito a retirada do erro da linha 39.
________________________________________

"""

# ----------------- INTERFACE DE CHAT INDIVIDUAL -----------------
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
    
    st.subheader("Dados para elaboração do recurso")
    
    input_counter = st.session_state.get(f"{model_key}_input_counter", 0)
    
    identificacao_key = f"{model_key}_identificacao_erros_{input_counter}"
    identificacao_erros = st.text_area(
        "Identificação dos erros:",
        key=identificacao_key,
        height=100,
        placeholder="Insira a identificação dos erros..."
    )
    
    texto_key = f"{model_key}_texto_analisado_{input_counter}"
    texto_analisado = st.text_area(
        "Texto a Ser Analisado:",
        key=texto_key,
        height=150,
        placeholder="Insira o texto a ser analisado..."
    )
    
    col1, col2, col3, col4, col5 = st.columns([1.5, 2, 2, 2, 2])
    with col1:
        if st.button("Enviar Mensagem", key=f"send_{model_key}"):
            if identificacao_erros.strip() and texto_analisado.strip():
                prompt_formatado = PROMPT_TEMPLATE.format(
                    identificacao_erros=identificacao_erros.strip(),
                    texto_analisado=texto_analisado.strip()
                )
                st.session_state[f"{model_key}_messages"].append({"role": "user", "content": prompt_formatado})
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
        rating = st.selectbox(
            "Avalie o modelo (1-5)",
            options=[1, 2, 3, 4, 5],
            index=0,
            key=f"rating_{model_key}_{rating_counter}",
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

# ----------------- INTERFACE GLOBAL – AVALIAÇÃO SIMULTÂNEA -----------------
def global_chat_interface():
    # Renomeada para "Avaliação simultânea"
    if "global_messages" not in st.session_state:
        st.session_state.global_messages = []
    with st.expander("Histórico de Conversa (Avaliação simultânea)", expanded=True):
        for msg in st.session_state.global_messages:
            if msg["role"] == "user":
                display_message("user", msg["content"])
            else:
                display_message("assistant", msg["content"], "Avaliação simultânea")
    input_counter = st.session_state.get("global_input_counter", 0)
    
    global_identificacao_key = f"global_identificacao_erros_{input_counter}"
    global_identificacao_erros = st.text_area(
        "Identificação dos erros:",
        key=global_identificacao_key,
        height=100,
        placeholder="Insira a identificação dos erros..."
    )
    
    global_texto_key = f"global_texto_analisado_{input_counter}"
    global_texto = st.text_area(
        "Texto a Ser Analisado:",
        key=global_texto_key,
        height=150,
        placeholder="Insira o texto a ser analisado..."
    )
    
    # Dicionário com os modelos disponíveis
    available_models = {
        "GPT-4o-Mini": ('gpt', analyze_with_gpt),
        "GPT-4": ('gpt4', analyze_with_gpt4),
        "Claude-3.7-Sonnet": ('claude', analyze_with_claude),
        "Maritalk Sabiá-3": ('maritalk', analyze_maritalk),
        "Llama 405B": ('llama405b', analyze_llama405b),
        "Deepseek Chat": ('deepseek_chat', analyze_deepseek_chat),
        "Deepseek Reasoner": ('deepseek_reasoner', analyze_deepseek_reasoner),
        "Gemini 2.0 Flash": ('gemini', analyze_with_gemini),
        "O3-Mini-high": ('o3-mini-high', analyze_with_o3_mini_high),
        "Qwen-Max": ('qwen-max', analyze_with_qwen_max),
        "Qwen-Plus": ('qwen-plus', analyze_with_qwen_plus)
    }
    
    # Caixa de seleção múltipla para escolha dos modelos
    selected_models = st.multiselect(
        "Selecione os modelos para enviar a mensagem:",
        options=list(available_models.keys()),
        default=list(available_models.keys())
    )
    
    col1, col4 = st.columns([1.5, 2])
    with col1:
        if st.button("Enviar Mensagem para os modelos selecionados", key="send_all"):
            if global_identificacao_erros.strip() and global_texto.strip():
                prompt_formatado = PROMPT_TEMPLATE.format(
                    identificacao_erros=global_identificacao_erros.strip(),
                    texto_analisado=global_texto.strip()
                )
                st.session_state.global_messages.append({"role": "user", "content": prompt_formatado})
                models_to_send = [ available_models[m] for m in selected_models ]
                for model_key, _ in models_to_send:
                    messages_key = f"{model_key}_messages"
                    if messages_key not in st.session_state:
                        st.session_state[messages_key] = []
                    st.session_state[messages_key].append({"role": "user", "content": prompt_formatado})
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
                    with st.spinner("Processando os modelos selecionados..."):
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

# ----------------- INTERFACE PRINCIPAL -----------------
def main_interface():
    st.title("Português - Aspectos Microestruturais")
    if "resultados" not in st.session_state:
        st.session_state["resultados"] = []
    tabs = st.tabs([
        "Avaliação simultânea",
        "GPT-4o-Mini",
        "GPT-4",
        "Claude-3.7-Sonnet",
        "Maritalk Sabiá-3",
        "Llama 405B",
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
        chat_interface('claude', 'Claude-3.7-Sonnet', analyze_with_claude)
    with tabs[4]:
        chat_interface('maritalk', 'Maritalk Sabiá-3', analyze_maritalk)
    with tabs[5]:
        chat_interface('llama405b', 'Llama 405B', analyze_llama405b)
    with tabs[6]:
        chat_interface('deepseek_chat', 'Deepseek Chat', analyze_deepseek_chat)
    with tabs[7]:
        chat_interface('deepseek_reasoner', 'Deepseek Reasoner', analyze_deepseek_reasoner)
    with tabs[8]:
        chat_interface('gemini', 'Gemini 2.0 Flash', analyze_with_gemini)
    with tabs[9]:
        chat_interface('o3-mini-high', 'O3-Mini-high', analyze_with_o3_mini_high)
    with tabs[10]:
        chat_interface('qwen-max', 'Qwen-Max', analyze_with_qwen_max)
    with tabs[11]:
        chat_interface('qwen-plus', 'Qwen-Plus', analyze_with_qwen_plus)
    with tabs[12]:
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
    with tabs[13]:
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
            fig_scatter = px.scatter(df_summary, x="Custo_num", y="nota obtida", size="interações", color="modelo",
                                     hover_name="modelo", title="Avaliação Média vs Custo Médio por Modelo")
            st.plotly_chart(fig_scatter, use_container_width=True)
        else:
            st.info("Nenhum dado disponível para gerar gráficos.")
    with tabs[14]:
        st.subheader("Texto do Recurso")
        texto_recurso = st.session_state.get("texto_recurso", "")
        st.text_area("Texto do Recurso:", value=texto_recurso, height=300)

# ----------------- INICIALIZAÇÃO -----------------
if "clients" not in st.session_state:
    st.session_state.clients = init_clients()
main_interface()
