FROM public.ecr.aws/lambda/python:3.11

# Working directory
WORKDIR /var/task

# System dependencies
# gcc/g++ needed by psycopg; mesa-libGL needed by OpenCV
RUN yum install -y \
        gcc \
        gcc-c++ \
        make \
        mesa-libGL \
        glib2 \
        libSM \
        libXrender \
        libXext && \
    yum clean all

# Install Python dependencies
COPY requirements.txt .
RUN pip install --upgrade pip --no-cache-dir && \
    pip install --no-cache-dir -r requirements.txt

# Copy Lambda source  (same pattern as your existing Dockerfile)
COPY lambda_function.py .
COPY lib/ ./lib/

# Lambda handler
CMD ["lambda_function.lambda_handler"]