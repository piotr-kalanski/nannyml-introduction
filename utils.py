import inspect
import os
import streamlit as st


def display_source_code():
    st.write("---")
    st.header('Source code')
    caller_path = os.path.abspath((inspect.stack()[1])[1])
    with open(caller_path, 'r') as f:
        code = ''.join(f.readlines()[:-2]) # ignore two last lines
    st.code(code, language='python')
