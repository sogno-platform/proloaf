FROM python:3
WORKDIR /usr/src/app
COPY ./requirements.txt ./
COPY ./setup.py ./
COPY ./README.md ./
COPY ./src ./src
RUN pip3 install .
RUN pip3 install --no-cache-dir -r requirements.txt