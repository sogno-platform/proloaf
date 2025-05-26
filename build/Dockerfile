FROM python:3
WORKDIR /usr/src/app
COPY ./pyproject.toml ./
COPY ./README.md ./
COPY ./src ./src
RUN pip install --upgrade pip
RUN pip3 install .