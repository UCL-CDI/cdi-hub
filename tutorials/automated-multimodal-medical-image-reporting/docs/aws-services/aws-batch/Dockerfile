FROM public.ecr.aws/amazoncorretto/amazoncorretto:11

# RUN yum install -y git python-pip curl jq
# RUN pip install --upgrade awscli

WORKDIR workspace
COPY scripts/entrypoint.bash /usr/local/bin/entrypoint.bash
RUN chmod +x /usr/local/bin/entrypoint.bash

#VOLUME ["/input-data"] ?might not need it?

CMD ["/usr/local/bin/entrypoint.bash"]
