FROM tensorflow/tensorflow@latest
WORKDIR api
COPY main.py main.py
COPY utils.py utils.py
COPY requirements.txt requirements.txt
RUN pip install -r requirements.txt
CMD uvicorn main:app