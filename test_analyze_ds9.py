import requests
import json

BASE_URL = "http://localhost:8000"

def test_analyze_ds9():
    project_id = 2
    dataset_id = 9
    
    # Check dataset columns
    resp = requests.get(f"{BASE_URL}/api/datasets/{dataset_id}")
    ds = resp.json()
    cols = ds.get('columns_info', {})
    print(f"Dataset cols count: {len(cols)}")
    print(f"Target column in DB: { {k:v for k,v in cols.items() if k in ['target', 'label']} }")
    
    # Build a simple rule if we find a numeric col
    # E.g. one of those FT2_... cols
    rules = []
    num_cols = [k for k, v in cols.items() if 'float' in v or 'int' in v]
    if num_cols:
        rules.append({"field": num_cols[0], "op": ">", "val": 0})
    
    payload = {
        "project_id": project_id,
        "dataset_id": dataset_id,
        "rules": rules,
        "combine_logic": "and"
    }
    
    print(f"Testing analyze with payload: {payload['project_id']}, {payload['dataset_id']}")
    resp = requests.post(f"{BASE_URL}/api/strategies/analyze", json=payload)
    print(f"Analyze Status: {resp.status_code}")
    if resp.status_code == 200:
        print(f"Result metrics: {resp.json().keys()}")
        print(f"Sample: { {k: resp.json()[k] for k in list(resp.json().keys())[:3]} }")
    else:
        print(f"Error Result: {resp.text}")

if __name__ == "__main__":
    test_analyze_ds9()
