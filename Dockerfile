# Student Early-Warning System (OULAD #99 + UCI Dropout #100)
FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
# Build dataset + model + demo DB at image-build time so `docker run` is instant
RUN python -m src.generate_data && python -m src.train_model && python -m src.database
EXPOSE 8501
CMD ["streamlit", "run", "app.py", "--server.port", "8501", "--server.address", "0.0.0.0", "--server.headless", "true"]
