import requests
import time

BASE_URL = "http://localhost:8000/api/v1"

def run_tests():
    # 1. 尝试使用 root 登录
    print("1. 测试 root 账号登录...")
    resp = requests.post(f"{BASE_URL}/auth/login", data={"username": "root", "password": "root"})
    if resp.status_code == 200:
        root_token = resp.json() ["access_token"]
        print("-> root 登录成功！")
    else:
        print(f"-> root 登录失败！{resp.text}")
        return

    # 2. 从 root 获取所有用户
    print("\n2. 测试 root 获取所有用户列表...")
    headers_root = {"Authorization": f"Bearer {root_token}"}
    resp = requests.get(f"{BASE_URL}/users/all", headers=headers_root)
    if resp.status_code == 200:
        print(f"-> 获取成功，当前用户列表数量: {len(resp.json())}")
    else:
        print(f"-> 获取全量用户失败！{resp.text}")

    # 3. 注册新用户
    print("\n3. 注册新用户 normal_user...")
    resp = requests.post(f"{BASE_URL}/users/register", json={"username": "normal_user", "password": "123"})
    if resp.status_code == 200:
        print("-> 注册成功！")
        normal_user_id = resp.json()["id"]
    else:
        print(f"-> 注册失败 (可能已存在)！响应: {resp.text}")
        # 如果存在，尝试登录获取 token 即可
        resp_login = requests.post(f"{BASE_URL}/auth/login", data={"username": "normal_user", "password": "123"})
        normal_user_id = resp_login.json()["user"]["id"]

    # 4. 普通用户登录
    print("\n4. 测试 normal_user 登录...")
    resp = requests.post(f"{BASE_URL}/auth/login", data={"username": "normal_user", "password": "123"})
    if resp.status_code == 200:
        normal_token = resp.json()["access_token"]
        print("-> normal_user 登录成功！")
    else:
        print(f"-> normal_user 登录失败！{resp.text}")
        return

    # 5. 测试越权访问
    print("\n5. 测试 normal_user 越权调用 get_all_users...")
    headers_normal = {"Authorization": f"Bearer {normal_token}"}
    resp = requests.get(f"{BASE_URL}/users/all", headers=headers_normal)
    if resp.status_code == 403:
        print("-> 越权防护成功！正常返回了 403。")
    else:
        print(f"-> 越权防护可能失效！返回码: {resp.status_code}, {resp.text}")

    # 6. root 提权 normal_user
    print("\n6. 测试 root 赋予 normal_user 管理员权限...")
    resp = requests.put(f"{BASE_URL}/users/{normal_user_id}/role", json={"is_admin": 1}, headers=headers_root)
    if resp.status_code == 200:
        print("-> 提权成功！")
    else:
        print(f"-> 提权失败！{resp.text}")

if __name__ == "__main__":
    run_tests()
