
FROM ghcr.io/ansible-community/community-ee-minimal:latest

USER root

ENV HOME=/root \
    ANSIBLE_HOME=/root/.ansible \
    ANSIBLE_LOCAL_TEMP=/tmp/.ansible/tmp

WORKDIR /workspace

RUN mkdir -p /tmp/.ansible/tmp && \
    printf '%s\n' \
    '#!/bin/sh' \
    'set -eu' \
    'mkdir -p /root/.ssh/known-hosts' \
    'chmod 700 /root/.ssh /root/.ssh/known-hosts' \
    'cp /keys/main.pem /root/.ssh/main.pem' \
    'chmod 600 /root/.ssh/main.pem' \
    'exec "$@"' \
    > /usr/local/bin/ansible-entrypoint.sh && \
    chmod 755 /usr/local/bin/ansible-entrypoint.sh

ENTRYPOINT ["/usr/local/bin/ansible-entrypoint.sh"]

CMD ["tail", "-f", "/dev/null"]
