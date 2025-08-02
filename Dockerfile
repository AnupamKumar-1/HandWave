# 1) Use official lightweight Python base image
FROM python:3.10-slim

# 2) Environment settings to prevent .pyc creation and enable stdout logging
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# 3) Install OS-level dependencies for Python packages and OpenCV
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# 4) Set the working directory
WORKDIR /usr/src/app

# 5) Install Python dependencies (including Gunicorn for production)
COPY requirements.txt .
RUN pip install --upgrade pip \
    && pip install -r requirements.txt

# 6) Copy the rest of the application code
COPY . .

# 7) Expose the port that Render will route to (via $PORT env var)
EXPOSE 5000

# 8) Default command to run the application with Gunicorn binding to $PORT
CMD ["gunicorn", "webapp.app:app", "--bind", "0.0.0.0:$PORT"]
