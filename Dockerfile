# ---- Base Python (CPU) ----
FROM python:3.12-slim-bookworm

# Avoid .pyc, ensure unbuffered logs
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Workdir inside container
WORKDIR /code

# System deps (optional but useful for Pillow, zip, etc.)
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 unzip && \
    rm -rf /var/lib/apt/lists/*

# ---- Python deps ----
COPY requirements.txt /code/requirements.txt

RUN python -m pip install --upgrade pip && \
    pip install --no-cache-dir -r /code/requirements.txt && \
    pip install --no-cache-dir \
        torch==2.2.2 torchvision==0.17.2 torchaudio==2.2.2 \
        --index-url https://download.pytorch.org/whl/cpu


# ---- App code ----
COPY app /code/app
COPY helper_lib /code/helper_lib
# weights are needed for inference endpoints (GAN generator)
COPY weights /code/weights

# Expose port and run
EXPOSE 8000
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "80"]
