FROM ubuntu:24.04

ENV DEBIAN_FRONTEND=noninteractive TZ=Etc/UTC

RUN apt-get update \
 && apt-get install -y --no-install-recommends \
      texlive-latex-base \
      texlive-latex-recommended \
      texlive-latex-extra \
      texlive-fonts-recommended \
      texlive-fonts-extra \
      texlive-science    \
      latexmk \
      biber \
      make \
 && apt-get clean \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /work

CMD ["bash"]
