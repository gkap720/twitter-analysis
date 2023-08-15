FROM pytorch/pytorch:latest
WORKDIR fastapi
COPY requirements.txt requirements.txt
RUN pip install -r requirements.txt
COPY src src
COPY setup.py setup.py
RUN pip install .
RUN python src/utils/nltk_setup.py
ENTRYPOINT ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "80"]