FROM cmeiklejohn/ml

ARG VERSION
RUN pip install --upgrade pip \
    && rm -rf ~/.cache/pip

RUN pip install flask

RUN pip install flasgger

# Add entrypoint (this allows variable expansion)
COPY entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

EXPOSE 5000

COPY ./apps/ /apps

# enforce version of important python modules
RUN pip install -r /apps/requirement.txt

#RUN echo "Flask Server logs" > /var/log/stdout.log

WORKDIR /

ENTRYPOINT ["sh", "-c", "/entrypoint.sh >>/var/log/stdout.log 2>>/var/log/stderr.log"]
