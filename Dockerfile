FROM quay.io/jupyter/minimal-notebook:python-3.11.6

COPY environment.yml /tmp/environment.yml

RUN mamba env create -f /tmp/environment.yml && \
    mamba clean -afy && \
    /opt/conda/envs/dsci_522_lab_env/bin/python -m ipykernel install --user \
      --name dsci_522_lab_env \
      --display-name "Python (dsci_522_lab_env)"
      
ENV CONDA_DEFAULT_ENV=dsci_522_lab_env
ENV PATH="/opt/conda/envs/dsci_522_lab_env/bin:$PATH"

WORKDIR /home/jovyan