# 1. 베이스 이미지 (Python + Linux)
FROM python:3.11-slim

# 2. 기본 패키지 업데이트 & 빌드 툴 설치
RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    wget \
    git \
    && rm -rf /var/lib/apt/lists/*

# 3. 작업 디렉터리 설정
WORKDIR /app

# 4. requirements 설치
# (먼저 requirements만 복사하면, 코드 바뀌어도 캐시 유지 가능)
COPY requirements.txt /app/requirements.txt

RUN pip install --no-cache-dir -r requirements.txt

# 5. 나머지 코드 복사
COPY . /app

# 6. 컨테이너 시작 시 uvicorn 실행 (포트 8000)
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]