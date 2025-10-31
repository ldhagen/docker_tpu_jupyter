FROM ubuntu:20.04

ENV DEBIAN_FRONTEND=noninteractive

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    wget \
    curl \
    gnupg \
    lsb-release \
    vim \
    git \
    && rm -rf /var/lib/apt/lists/*

# Add Google Coral repository
RUN echo "deb https://packages.cloud.google.com/apt coral-edgetpu-stable main" | tee /etc/apt/sources.list.d/coral-edgetpu.list
RUN curl https://packages.cloud.google.com/apt/doc/apt-key.gpg | apt-key add -

# Install Edge TPU runtime
RUN apt-get update && apt-get install -y \
    libedgetpu1-std \
    python3-pycoral \
    && rm -rf /var/lib/apt/lists/*

# Install Python packages
RUN pip3 install --upgrade pip
RUN pip3 install tensorflow==2.13.0
RUN pip3 install tflite-runtime
RUN pip3 install numpy pillow matplotlib jupyter

# Create workspace
RUN mkdir -p /workspace
WORKDIR /workspace

# Download test model with multiple fallback URLs
RUN wget -q https://github.com/google-coral/test_data/raw/master/mobilenet_v1_1.0_224_quant_edgetpu.tflite -O mobilenet_edgetpu.tflite || \
    wget -q https://raw.githubusercontent.com/google-coral/test_data/master/mobilenet_v1_1.0_224_quant_edgetpu.tflite -O mobilenet_edgetpu.tflite || \
    wget -q https://storage.googleapis.com/mobilenet_v1_1.0_224_quant_edgetpu.tflite -O mobilenet_edgetpu.tflite || \
    echo "Model download failed - will download manually"

# Copy test scripts
COPY *.py ./

# Simple startup script
RUN echo '#!/bin/bash\n\
echo "=== Edge TPU Development Environment ==="\n\
echo "Python: $(python3 --version)"\n\
\n\
# Download model if missing\n\
if [ ! -f "mobilenet_edgetpu.tflite" ]; then\n\
    echo "Downloading test model..."\n\
    wget -q https://github.com/google-coral/test_data/raw/master/mobilenet_v1_1.0_224_quant_edgetpu.tflite -O mobilenet_edgetpu.tflite || \\\n\
    wget -q https://raw.githubusercontent.com/google-coral/test_data/master/mobilenet_v1_1.0_224_quant_edgetpu.tflite -O mobilenet_edgetpu.tflite || \\\n\
    echo "Model download failed"\n\
fi\n\
\n\
if [ -f "test_edgetpu.py" ]; then\n\
    python3 test_edgetpu.py\n\
fi\n\
\n\
echo ""\n\
echo "Starting shell..."\n\
/bin/bash' > /start.sh

RUN chmod +x /start.sh

CMD ["/start.sh"]
