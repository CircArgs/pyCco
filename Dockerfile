FROM arm32v7/ubuntu

# Set up a working directory
WORKDIR /workspace

# Install essential tools for assembling and linking ARM assembly
RUN apt-get update && apt-get install -y \
    binutils \
    gcc \
    build-essential \
    && apt-get clean

# Set up entrypoint to start with a bash shell
CMD ["/bin/bash"]
