FROM felddy/weewx:5
USER root

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        usbutils \
        libusb-1.0-0-dev && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

RUN /opt/venv/bin/pip install --no-cache-dir pyusb paho-mqtt

RUN mkdir -p /data/bin/user

ENTRYPOINT ["weewxd"]
CMD ["/data/weewx.conf"]
