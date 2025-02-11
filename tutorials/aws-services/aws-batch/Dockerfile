FROM python:slim

RUN apt-get update 
RUN apt-get install -y git curl python3 python3-pip
RUN apt install -y vim
RUN pip install --upgrade awscli

WORKDIR workspace

COPY scripts/entrypoint.bash .
RUN chmod +x entrypoint.bash

#?COPY scripts/entrypoint.bash /usr/local/bin/entrypoint.bash
#?RUN chmod +x /usr/local/bin/entrypoint.bash
#?CMD ["/usr/local/bin/entrypoint.bash"]

ENTRYPOINT ["tail", "-f", "/dev/null"]

SHELL ["/bin/bash", "-cu"]