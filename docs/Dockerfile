FROM fedora:30

RUN dnf -y update

RUN dnf -y install \
  pandoc \
  wget \
  git \
  nodejs

WORKDIR /hugo
RUN wget https://github.com/gohugoio/hugo/releases/download/v0.71.0/hugo_extended_0.71.0_Linux-64bit.tar.gz && \
  tar -xvf hugo_extended_0.71.0_Linux-64bit.tar.gz && \
  install hugo /usr/bin

WORKDIR /website
RUN  npm install -D --save autoprefixer && \
  npm install -D -g --save postcss-cli
  # npm audit fix

EXPOSE 1313