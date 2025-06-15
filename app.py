import json
import boto3
import joblib
import os
import numpy as np

s3 = boto3.client('s3')
bucket_name = 'escomcareer-model'

# Mapeo de modelos y scalers por tipo
modelos_info = {
    'aptitudes': {
        'model_key': 'ModelRepository/modelo_random_forest_aptitudes_optimizado.pkl',
        'model_path': '/tmp/modelo_random_forest_aptitudes_optimizado.pkl',
        'scaler_key': 'ModelRepository/scaler_aptitudes.pkl',
        'scaler_path': '/tmp/scaler_aptitudes.pkl'
    },
    'intereses': {
        'model_key': 'ModelRepository/modelo_random_forest_Intereses_optimizado.pkl',
        'model_path': '/tmp/modelo_random_forest_Intereses_optimizado.pkl',
        'scaler_key': 'ModelRepository/scaler_intereses.pkl',
        'scaler_path': '/tmp/scaler_intereses.pkl'
    }
}

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

    # Manejo de preflight CORS
    if event.get('httpMethod') == 'OPTIONS':
        return {
            'statusCode': 200,
            'headers': cors_headers,
            'body': ''
        }

    try:
        # Detectar la ruta
        path = event.get('path', '')
        if path.endswith('/aptitudes'):
            tipo = 'aptitudes'
        elif path.endswith('/intereses'):
            tipo = 'intereses'
        else:
            raise ValueError("Ruta no válida. Usa /aptitudes o /intereses.")

        # Parsear el cuerpo
        if 'body' in event:
            body = json.loads(event['body'])
        else:
            body = event

        features = body.get('features')
        if not features or not isinstance(features, list):
            raise ValueError("Debes proporcionar un arreglo de características en 'features'.")

        # Cargar modelo y scaler según el tipo
        info = modelos_info[tipo]
        download_file_if_not_exists(info['model_key'], info['model_path'])
        download_file_if_not_exists(info['scaler_key'], info['scaler_path'])

        model = joblib.load(info['model_path'])
        scaler = joblib.load(info['scaler_path'])

        # Preprocesamiento
        features = np.array(features).reshape(1, -1)
        features_normalized = scaler.transform(features)

        # Predicción
        prediction = model.predict(features_normalized)[0]
        proba = model.predict_proba(features_normalized)[0]
        probabilidades = {str(clase): float(prob) for clase, prob in zip(model.classes_, proba)}

        return {
            'statusCode': 200,
            'headers': cors_headers,
            'body': json.dumps({
                'tipo': tipo,
                'prediccion': str(prediction),
                'probabilidades': probabilidades
            })
        }

    except Exception as e:
        return {
            'statusCode': 500,
            'headers': cors_headers,
            'body': json.dumps({'error': str(e)})
        }
