import requests
import json
import time

BASE_URL = "http://localhost:8000"

def test_api():
    print("--- 1. Testing Project API ---")
    resp = requests.get(f"{BASE_URL}/api/projects")
    data = resp.json()
    projects = data.get('items', [])
    print(f"Total Projects: {len(projects)}")
    
    # Try to find Project 1
    project = next((p for p in projects if p['id'] == 1), projects[0] if projects else None)
    if not project:
        print("No project found!")
        return
    
    project_id = project['id']
    print(f"Using Project ID: {project_id}")

    print("\n--- 2. Testing Dataset API ---")
    resp = requests.get(f"{BASE_URL}/api/projects/{project_id}/datasets")
    datasets = resp.json()
    print(f"Total Datasets: {len(datasets)}")
    
    dataset = next((d for d in datasets if d['id'] == 1), datasets[0] if datasets else None)
    if not dataset:
        print("No dataset found!")
        return
    
    dataset_id = dataset['id']
    print(f"Using Dataset ID: {dataset_id}")
    
    resp = requests.get(f"{BASE_URL}/api/datasets/{dataset_id}")
    ds_detail = resp.json()
    print(f"Dataset Name: {ds_detail.get('name')}")
    print(f"Dataset Stats Cache Keys: {list(ds_detail.get('stats_cache', {}).keys())}")
    
    print("\n--- 3. Testing IV Report (V2 Feature) ---")
    cols = ds_detail.get('columns_info', {})
    target = 'label'
    features = [c for c in cols.keys() if c not in [target, 'weight']][:10]
    
    payload = {
        "dataset_id": dataset_id,
        "dep": target,
        "features": features,
        "split_ratios": [0.6, 0.2, 0.2],
        "exclude_cols": []
    }
    
    # Corrected route
    iv_url = f"{BASE_URL}/api/projects/{project_id}/feature/iv-report"
    print(f"Posting to: {iv_url}")
    resp = requests.post(iv_url, json=payload)
    print(f"IV Report Status: {resp.status_code}")
    if resp.status_code == 200:
        report = resp.json()
        ft_results = report.get('features', [])
        print(f"Report Features: {len(ft_results)}")
        if ft_results:
            print(f"Sample Feature Metric: {ft_results[0]}")

    print("\n--- 4. Testing Model Deployment & Simulation ---")
    resp = requests.get(f"{BASE_URL}/api/projects/{project_id}/results")
    results = resp.json()
    items = results.get('items', [])
    print(f"Model Results: {len(items)}")
    if items:
        model_id = items[0]['id']
        print(f"Deploying Model Result: {model_id}")
        
        resp = requests.post(f"{BASE_URL}/api/v1/deploy-model/{project_id}/{model_id}")
        print(f"Deploy Response: {resp.json()}")
        
        deployment_id = resp.json().get('id')
        if deployment_id:
            print(f"\n--- 5. Testing Monitoring Simulation ---")
            sim_payload = {
                "deployment_id": deployment_id,
                "n_samples": 500,
                "drift_scale": 0.5, # Increase drift to see PSI > 0
                "batch_name": "Antigravity Test Batch V2"
            }
            resp = requests.post(f"{BASE_URL}/api/monitor/simulate", json=sim_payload)
            print(f"Simulation Status: {resp.status_code}")
            if resp.status_code == 200:
                log = resp.json()
                print(f"Simulation Log - PSI: {log.get('psi')}, Avg Score: {log.get('avg_score')}")
    else:
        print("No model results found to deploy.")

if __name__ == "__main__":
    try:
        test_api()
    except Exception as e:
        import traceback
        traceback.print_exc()
