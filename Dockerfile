FROM python:3.10.6

COPY requirements.txt requirements.txt
RUN pip install --upgrade pip
RUN pip install -r requirements.txt

COPY package_folder package_folder
COPY raw_data raw_data
COPY models models

CMD uvicorn package_folder.api_file:app --host 0.0.0.0 --port $PORT
