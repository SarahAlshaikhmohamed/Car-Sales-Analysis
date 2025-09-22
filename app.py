import os
import subprocess

# Start FastAPI in background
subprocess.Popen(
    ["uvicorn", "Model.Interface:app", "--host", "0.0.0.0", "--port", "8000"]
)

# Run Streamlit (dashboard)
os.system("streamlit run Analytics/Dashboard.py --server.port 7860 --server.address 0.0.0.0")
