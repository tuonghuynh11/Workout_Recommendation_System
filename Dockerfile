FROM python:3.12.1

RUN mkdir -p app/backend

COPY requirements.txt app/backend/requirements.txt

COPY . app/backend

WORKDIR /app/backend

RUN pip install --upgrade pip
RUN pip install -r requirements.txt

EXPOSE 8080

CMD ["uvicorn","main:app","--host","0.0.0.0","--port","8080","--reload"]

