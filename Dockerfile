FROM python:3.9

RUN apt-get update && apt-get install -y xvfb libgl1-mesa-dev && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

ENTRYPOINT ["python"]
