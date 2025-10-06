# Use a lightweight Python image
FROM python:3.10-slim

# Set working directory
WORKDIR /code

# Copy requirements first (for caching)
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy all source code
COPY . .

# Expose the port Hugging Face expects
EXPOSE 7860

# Run the FastAPI app
CMD ["uvicorn", "app.Interface:app", "--host", "0.0.0.0", "--port", "7860"]