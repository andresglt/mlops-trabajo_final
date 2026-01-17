# Imagen base ligera de Python
FROM python:3.11-slim

# Directorio de trabajo dentro del contenedor
WORKDIR /app

# Copiar primero requirements para aprovechar caché
COPY serving/requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copiar todo el repositorio (incluye serving, data, models, dvc.yaml, etc.)
COPY . /app

# Asegurar que Python vea /app como parte del PYTHONPATH
ENV PYTHONPATH=/app

# Configurar puerto
ENV PORT=8000
EXPOSE 8000

# Comando de arranque: Uvicorn con FastAPI
CMD ["uvicorn", "serving.app:app", "--host", "0.0.0.0", "--port", "8000"]