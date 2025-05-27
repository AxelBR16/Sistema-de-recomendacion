import json
import boto3
import joblib
import os
import numpy as np

s3 = boto3.client('s3')
bucket_name = 'escomcareer-model'

model_key = 'ModelRepository/modelo_random_forest_aptitudes_optimizado.pkl'
model_path = '/tmp/modelo_random_forest_aptitudes_optimizado.pkl'

scaler_key = 'ModelRepository/scaler_aptitudes.pkl'
scaler_path = '/tmp/scaler_aptitudes.pkl'

def download_file_if_not_exists(s3_key, local_path):
    if not os.path.exists(local_path):
        s3.download_file(bucket_name, s3_key, local_path)

def lambda_handler(event, context):
    cors_headers = {
        'Access-Control-Allow-Origin': '*',
        'Access-Control-Allow-Headers': 'Content-Type,X-Amz-Date,Authorization,X-Api-Key,X-Amz-Security-Token',
        'Access-Control-Allow-Methods': 'GET,POST,OPTIONS',
        'Content-Type': 'application/json'
    }
    
    if event.get('httpMethod') == 'OPTIONS':
        return {
            'statusCode': 200,
            'headers': cors_headers,
            'body': ''
        }
    
    try:
        # Descargar modelo y scaler
        download_file_if_not_exists(model_key, model_path)
        download_file_if_not_exists(scaler_key, scaler_path)
        
        model = joblib.load(model_path)
        scaler = joblib.load(scaler_path)
        
        if 'body' in event:
            body = json.loads(event['body'])
        else:
            body = event
        
        features = np.array(body['features']).reshape(1, -1)

        # Normalizar características
        features_normalized = scaler.transform(features)
        
        # Realizar predicción
        prediction = model.predict(features_normalized)[0]
        proba = model.predict_proba(features_normalized)[0]
        
        probabilidades = {str(clase): float(prob) for clase, prob in zip(model.classes_, proba)}
        
        return {
            'statusCode': 200,
            'headers': cors_headers,
            'body': json.dumps({
                'prediccion': str(prediction),
                'probabilidades': probabilidades
            })
        }
        
    except Exception as e:
        return {
            'statusCode': 500,
            'headers': cors_headers,
            'body': json.dumps({
                'error': str(e)
            })
        }
