# Base image
FROM python:3.10-slim

# Set working directory
WORKDIR /code

# Copy project files
COPY . /code/

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# ✅ Create writable directories for Streamlit and Matplotlib
RUN mkdir -p /tmp/.streamlit /tmp/matplotlib /tmp/.cache && chmod -R 777 /tmp

# ✅ Environment variables for Streamlit + Matplotlib
ENV STREAMLIT_CONFIG_DIR=/tmp/.streamlit
ENV STREAMLIT_CACHE_DIR=/tmp/.streamlit/cache
ENV MPLCONFIGDIR=/tmp/matplotlib
ENV XDG_CACHE_HOME=/tmp/.cache

# ✅ Optional (makes Streamlit quieter)
ENV PYTHONUNBUFFERED=1
ENV STREAMLIT_SERVER_HEADLESS=true
ENV STREAMLIT_SERVER_PORT=7860
ENV STREAMLIT_SERVER_ENABLECORS=false

# ✅ Expose ports (FastAPI + Streamlit)
EXPOSE 8000
EXPOSE 7860

# ✅ Run both FastAPI & Streamlit together
CMD bash -c "\
uvicorn app.app:app --host 0.0.0.0 --port 8000 & \
sleep 3 && \
streamlit run Analytics/Dashboard.py --server.port 7860 --server.address 0.0.0.0 --server.enableCORS false --browser.serverAddress 0.0.0.0"
