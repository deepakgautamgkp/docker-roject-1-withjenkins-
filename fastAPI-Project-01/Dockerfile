# Dockerfile

FROM python:3.11-slim

#set a working directory
WORKDIR /app

# system deps. Install build-essential if you need to complie wheels (Optional)
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
       ca-certificates \
       build-essential \
    && rm -rf /var/lib/apt/lists/*

# create a non-rrot user
ENV APP_USER=appuser
RUN useradd -m -s /bin/bash $APP_USER

#copy requirements & install
COPY app/requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

# copy application code
COPY app /app/app

#chown to non-root
RUN chown -R $APP_USER:$APP_USER /app
USER $APP_USER

# environment variables
ENV PYTHONUNBUFFERED=1
ENV PORT=80

# expose port
EXPOSE 80

# Final command: gunicorn with Uvicorn worker for async support
# -w: number of workers (adjust based on CPU; default 4)
# --threads if you want threads; we use async workers so threads not needed
CMD ["gunicorn", "-k", "uvicorn.workers.UvicornWorker", "app.main:app", "-w", "4", "--bind", "0.0.0.0:80", "--log-level", "info", "--access-logfile", "-"]
