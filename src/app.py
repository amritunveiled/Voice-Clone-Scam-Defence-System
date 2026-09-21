from pathlib import Path
import json
import streamlit as st

ROOT=Path(__file__).resolve().parents[1]
st.set_page_config(page_title="Voice Clone Scam Defense", layout="wide")
st.title("Voice Clone Scam Defense")
st.caption("15-day research prototype — Day 1 clean restart")
state=(ROOT/"PROJECT_STATE.md").read_text(encoding="utf-8")
st.subheader("Current project state")
st.code(state, language="text")
res=ROOT/"results/tables/day1_rtf.json"
if res.exists():
    st.subheader("Day 1 benchmark")
    st.json(json.loads(res.read_text(encoding="utf-8")))
else:
    st.info("Day 1 benchmarks have not been run in this fresh restart yet.")
