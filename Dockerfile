FROM python:3.11-slim

USER root

# 时区设置
RUN apt-get update && apt-get install -y --no-install-recommends tzdata && \
    ln -fs /usr/share/zoneinfo/Asia/Shanghai /etc/localtime && \
    echo "Asia/Shanghai" > /etc/timezone && \
    rm -rf /var/lib/apt/lists/*

# 设置环境变量
ENV TZ=Asia/Shanghai
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

WORKDIR /app

# 先装依赖，利用缓存
COPY requirements.txt .
RUN pip3 install --no-cache-dir -r requirements.txt -i https://mirrors.tuna.tsinghua.edu.cn/pypi/web/simple

# Linux 原生运行库。
COPY libUtils.so /usr/lib/
RUN ldconfig /usr/local/lib

# 复制项目代码
COPY . .

ENTRYPOINT ["python", "main.py"]
CMD ["--scenario", "launch20/easy/E01", "--red-policy", "r8_satellite_packages", "--blue-policy", "b3_min_cost_assignment", "--output-dir", "results/runs/demo", "--render-mode", "none", "--disable-log-color", "--enable-config", "--enable-state", "--enable-event"]
