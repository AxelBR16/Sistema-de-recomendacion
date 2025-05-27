FROM public.ecr.aws/lambda/python:3.10

# Copiar el código fuente
COPY app.py ${LAMBDA_TASK_ROOT}

# Instalar las librerías necesarias
RUN pip install --upgrade pip
RUN pip install numpy scikit-learn joblib boto3

# Establecer el handler Lambda
CMD ["app.lambda_handler"]
