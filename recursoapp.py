import streamlit as st

# --- PAGE SETUP ---
redacao_view = st.Page(
    "views/cesperedação.py",
    title="Redação - Aspectos Macroestruturais",
    icon=":material/account_circle:",
    default=True,
)
cespemicro_view = st.Page(
    "views/cespemicro.py",
    title="Português - Aspectos Microestruturais",   
    icon=":material/bar_chart:",
)

FGV_view = st.Page(
    "views/FGV.py",
    title="FGV",   
    icon=":material/book:",
)

# --- NAVIGATION SETUP --- (agrupando ambas as páginas em uma única seção)
pg = st.navigation(
    {
        "CEBRASPE - Ambiente de Validação": [redacao_view, cespemicro_view],
        "FGV - Ambiente de Validação": [FGV_view],
    }
)

# --- NAVIGATION SETUP [WITH SECTIONS]---
#pg = st.navigation(
 #   {
 #       "CEBRASPE - Ambiente de Validação1": [redação_view],
  #      "CEBRASPE - Ambiente de Validação2": [cespemicro_view],
   # }
#)


# --- SHARED ON ALL PAGES ---
st.sidebar.markdown("Recurso Aprovado")

# --- RUN NAVIGATION ---
pg.run()
