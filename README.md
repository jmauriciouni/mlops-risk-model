# mlops-risk-model

## Paso 1:

### Inicialización de DVC

dvc init
git add .dvc .dvcignore .gitignore
git commit -m "Init DVC"

### Opcional: (local storage)

dvc remote add -d local_storage ./dvc_storage
dvc push

### 1) Instalar dependencias
pip install -r requirements.txt

### 2) Ejecutar todo el pipeline con DVC
dvc repro

### Guardar versiones de data/model en remoto DVC (si configuraste uno)
dvc push

### Ver el historial de runs en MLflow
mlflow ui --backend-store-uri mlruns
