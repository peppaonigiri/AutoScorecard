import yaml
import os
import socket

def get_lan_ip():
    """获取本机局域网 IP"""
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        # 不需要真正连接，只是为了获取本机地址
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
        s.close()
        return ip
    except Exception:
        return "127.0.0.1"

def sync():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    config_path = os.path.join(base_dir, "config.yaml")
    env_path = os.path.join(base_dir, "frontend", ".env.local")

    if not os.path.exists(config_path):
        print("Error: config.yaml not found!")
        return

    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    # 获取后端监听配置
    host = config["server"]["host"]
    port = config["server"]["port"]

    # 确定前端连接后端的正确 IP
    # 如果后端监听 0.0.0.0，前端应该尝试连接该电脑的局域网 IP
    # 如果后端监听 127.0.0.1，前端连接 127.0.0.1
    target_ip = host
    if host == "0.0.0.0":
        target_ip = get_lan_ip()

    backend_url = f"http://{target_ip}:{port}"

    with open(env_path, "w", encoding="utf-8") as f:
        f.write(f"VITE_API_URL={backend_url}\n")
    
    print(f"Successfully synced config.yaml to frontend/.env.local")
    print(f"Frontend will now point to: {backend_url}")

if __name__ == "__main__":
    sync()
