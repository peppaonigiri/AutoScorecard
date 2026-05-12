packages = [
    'fastapi', 'uvicorn', 'sqlalchemy', 'pandas',
    'numpy', 'sklearn', 'xgboost', 'lightgbm',
    'optuna', 'toad', 'jose', 'bcrypt', 'joblib', 'yaml'
]
results = []
for pkg in packages:
    try:
        __import__(pkg)
        results.append(f'OK: {pkg}')
    except ImportError as e:
        results.append(f'MISSING: {pkg} - {e}')
print('\n'.join(results))
