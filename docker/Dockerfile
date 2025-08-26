# Use the official dolfinx image which includes dolfinx and petsc
FROM dolfinx/dolfinx:stable

# Set a working directory
WORKDIR /home/fenics

# Install extra tools: gmsh, meshio, mpi4py, and utilities
RUN apt-get update && apt-get install -y --no-install-recommends \
    gmsh \
    python3-pip \
    && rm -rf /var/lib/apt/lists/*

# Copy requirement file and install Python dependencies
COPY requirements.txt /home/fenics/requirements.txt
RUN pip3 install --no-cache-dir -r /home/fenics/requirements.txt

# Copy entrypoint
COPY entrypoint.sh /usr/local/bin/entrypoint.sh
RUN chmod +x /usr/local/bin/entrypoint.sh

ENTRYPOINT ["/usr/local/bin/entrypoint.sh"]
CMD ["/bin/bash"]