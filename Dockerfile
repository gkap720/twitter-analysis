FROM python:3.8.12-buster
WORKDIR fastapi
COPY requirements.txt requirements.txt
RUN pip install --no-cache-dir -r requirements.txt 
COPY src src
COPY models models
COPY setup.py setup.py
RUN pip install .
RUN python src/utils/nltk_setup.py
ENTRYPOINT ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "80"]