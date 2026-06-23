"""Root entry point for Streamlit Cloud deployment."""
import runpy
import os
import sys

# Point to the real Streamlit app
sys.path.insert(0, os.path.dirname(__file__))
runpy.run_path("streamlit_app/app.py", run_name="__main__")