"""Minimal Streamlit UI for the ToS semantic search service."""

import os

import requests
import streamlit as st

API = os.environ.get("API_URL", "http://localhost:8000")

st.set_page_config(page_title="ToS Search", layout="centered")
st.title("Semantic Similarity Search")
st.caption("Terms of Service clause search and unfairness detection")

text = st.text_area(
    "Query",
    "we may terminate your account at any time without notice",
    height=80,
)

top_k = st.slider("Results", 1, 20, 5)

if st.button("Search", type="primary"):
    with st.spinner("Searching..."):
        try:
            r = requests.post(
                f"{API}/analyze",
                json={"text": text, "top_k": top_k},
                timeout=30,
            )
            r.raise_for_status()
            data = r.json()
        except requests.exceptions.RequestException as e:
            st.error(f"API unavailable: {e}")
            st.stop()

        st.subheader("Classification")
        if data["classification"]["is_unfair"]:
            st.warning("Potentially unfair")
        else:
            st.success("No unfairness detected")
        for cat in data["classification"]["unfair_categories"]:
            if cat["predicted"]:
                st.markdown(f"🔴 **{cat['name']}** ({cat['confidence']:.2f})")
            else:
                st.markdown(f"🟢 {cat['name']} ({cat['confidence']:.2f})")

        st.subheader("Similar Clauses")
        for hit in data["similar_sentences"]:
            st.markdown(f"`{hit['score']:.3f}` {hit['sentence']}")