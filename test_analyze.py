import requests
import json

BASE_URL = "http://localhost:8000"

def test_analyze():
    project_id = 1
    dataset_id = 1
    
    # Check dataset columns first
    resp = requests.get(f"{BASE_URL}/api/datasets/{dataset_id}")
    ds = resp.json()
    cols = ds.get('columns_info', {})
    print(f"Dataset cols: {list(cols.keys())}")
    
    # Build rules for backtest
    # Using 'age' if it exists
    rules = []
    if 'age' in cols:
        rules.append({"field": "age", "op": ">", "val": 30})
    
    payload = {
        "project_id": project_id,
        "dataset_id": dataset_id,
        "rules": rules,
        "combine_logic": "and"
    }
    
    print(f"Testing analyze with payload: {payload}")
    resp = requests.post(f"{BASE_URL}/api/strategies/analyze", json=payload)
    print(f"Analyze Status: {resp.status_code}")
    if resp.status_code == 200:
        print(f"Result: {json.dumps(resp.json(), indent=2)}")
    else:
        print(f"Error Result: {resp.text}")

if __name__ == "__main__":
    test_analyze()
