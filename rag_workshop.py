import streamlit as st

st.title("Meine erste App")

name = st.text_input("Wie heißt du?")

if name:
    st.write(f"Hallo {name}!")