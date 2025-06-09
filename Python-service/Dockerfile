FROM python:3.11-slim

WORKDIR /app

# Copia os arquivos de dependências
COPY requirements.txt .

# Instala as dependências
RUN pip install --no-cache-dir -r requirements.txt

# Copia o código da aplicação
COPY . .

# Expõe a porta 5000
EXPOSE 5000

# Comando para iniciar a aplicação
CMD ["python", "recommendation_api.py"]