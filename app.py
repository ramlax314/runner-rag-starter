# placeholder - will be overwritten later
import os
import pandas as pd
import streamlit as st
from openai import OpenAI
from retriever import retrieve_chunks
from build_kb import collection_count, rebuild_kb

st.set_page_config(page_title="Runner RAG — Chat with Your Runs", page_icon="🏃", layout="wide")
st.title("🏃 Runner RAG — Chat with Your Runs")

def get_api_key():
    key = st.sidebar.text_input("OpenAI API Key", type="password", help="Paste your key or set OPENAI_API_KEY.")
    if key: return key.strip()
    if "OPENAI_API_KEY" in st.secrets: return st.secrets["OPENAI_API_KEY"]
    return os.getenv("OPENAI_API_KEY","").strip()

api_key = get_api_key()
if not api_key:
    st.info("Add your OpenAI API key in the sidebar, or via .streamlit/secrets.toml / env var.")

st.sidebar.header("⚙️ Settings")
model = st.sidebar.text_input("Model", value="gpt-4o-mini")
max_tokens = st.sidebar.slider("Max output tokens", 100, 1200, 500, step=50)

st.sidebar.markdown("### Knowledge Base")
kb_count = collection_count()
st.sidebar.write(f"KB chunks: **{kb_count}**")
if st.sidebar.button("Rebuild KB now"):
    if not api_key:
        st.error("No API key set.")
    else:
        n = rebuild_kb()
        st.sidebar.success(f"Rebuilt KB with {n} chunks.")

@st.cache_data
def load_runs(path: str) -> pd.DataFrame:
    df = pd.read_excel(path, sheet_name=0)
    if "Date" in df.columns:
        df["Date"] = pd.to_datetime(df["Date"]).dt.date
    return df

df = load_runs("data/runs.xlsx")
st.subheader("📊 Runs")
st.dataframe(df, use_container_width=True)
c1, c2, c3 = st.columns(3)
with c1: st.metric("Total km", f"{df['Distance_km'].sum():.1f}")
with c2: st.metric("Avg pace (min/km)", f"{df['Avg_Pace_min_per_km'].mean():.2f}")
with c3: st.metric("Avg HR (bpm)", f"{df['Avg_HR'].mean():.0f}")

def summarize_runs(df: pd.DataFrame) -> str:
    total_km = df["Distance_km"].sum()
    avg_pace = df["Avg_Pace_min_per_km"].mean()
    avg_hr = df["Avg_HR"].mean()
    fastest = df.loc[df["Avg_Pace_min_per_km"].idxmin()]
    slowest = df.loc[df["Avg_Pace_min_per_km"].idxmax()]
    lines = [
        f"Total distance: {total_km:.1f} km across {len(df)} runs.",
        f"Avg pace: {avg_pace:.2f} min/km | Avg HR: {avg_hr:.0f} bpm.",
        f"Fastest: {fastest['Distance_km']} km @ {fastest['Avg_Pace_min_per_km']:.2f} on {fastest['Date']}.",
        f"Slowest: {slowest['Distance_km']} km @ {slowest['Avg_Pace_min_per_km']:.2f} on {slowest['Date']}.",
    ]
    return "\n".join(lines)

st.subheader("💬 Chat")
user_q = st.text_area("Ask about your runs or training", height=100)
do_stream = st.toggle("Stream output", value=True)

if "chat" not in st.session_state:
    st.session_state.chat = []

def build_prompt(question: str, kb_chunks_text: str) -> str:
    return f"""
You are a supportive running coach and data analyst. Use the RUNS SUMMARY and RETRIEVED KNOWLEDGE to answer clearly.
Tie advice to the runs provided. Keep bullets concise. This is general guidance, not medical advice.

[RUNS SUMMARY]
{summarize_runs(df)}

[RETRIEVED KNOWLEDGE]
{kb_chunks_text}

[CONVERSATION]
{"".join([f"{m['role'].capitalize()}: {m['content']}\n" for m in st.session_state.chat[-6:]])}

[QUESTION]
{question}

[ANSWER]
""".strip()

if st.button("Ask", type="primary"):
    if not user_q.strip():
        st.warning("Please ask a question.")
        st.stop()
    if not api_key:
        st.error("No API key found.")
        st.stop()

    if collection_count() == 0:
        n = rebuild_kb()
        st.sidebar.success(f"Auto-built KB with {n} chunks.")

    topk = retrieve_chunks(user_q, k=3)
    kb_text = "\n\n".join([c for c,_ in topk]) if topk else "(no context found)"
    prompt = build_prompt(user_q, kb_text)

    client = OpenAI(api_key=api_key)
    st.markdown("**Assistant:**")
    if do_stream:
        def gen():
            with client.responses.stream(model=model, input=prompt, max_output_tokens=max_tokens) as stream:
                for event in stream:
                    if event.type == "response.output_text.delta":
                        yield event.delta
                    elif event.type == "response.completed":
                        break
        out = st.write_stream(gen())
    else:
        resp = client.responses.create(model=model, input=prompt, max_output_tokens=max_tokens)
        out = resp.output_text
        st.write(out)

    st.session_state.chat.append({"role":"user","content":user_q})
    st.session_state.chat.append({"role":"assistant","content":out})

st.caption("Sidebar → Rebuild KB after you edit files in /knowledge.")
