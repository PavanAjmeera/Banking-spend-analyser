import streamlit as st
import pandas as pd

st.title("Hello Hackathon v2")
df = pd.DataFrame({"month": ["Jan", "Feb", "Mar"], "spend": [8000, 9500, 7200]})
st.bar_chart(df.set_index("month"))