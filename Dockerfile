FROM python:3
WORKDIR /usr/src/app
COPY requirements.txt ./
COPY plf_util ./plf_util/
COPY setup.py ./
RUN ls
RUN python3 setup.py build
RUN python3 setup.py install
RUN pip3 install --no-cache-dir -r requirements.txt
# as long as plf_util is not available via pip, use following two lines
# COPY plf_util ./
# RUN pip3 install ./plf_util
