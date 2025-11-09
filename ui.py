# components/ui.py
import streamlit as st
from datetime import datetime

BRAND_BLOCK = """
**Developed by Mahbub Hassan**  
Department of Civil Engineering, Faculty of Engineering, Chulalongkorn University  
Founder, B'Deshi Emerging Research Lab  

Email: mahbub.hassan@ieee.org
"""

def header(title: str, subtitle: str | None = None):
    st.markdown(f"# {title}")
    if subtitle:
        st.markdown(f"**{subtitle}**")
    st.markdown("---")

def footer_brand():
    st.markdown("---")
    st.caption(BRAND_BLOCK)
    st.caption(f"© {datetime.now().year} · All rights reserved.")
