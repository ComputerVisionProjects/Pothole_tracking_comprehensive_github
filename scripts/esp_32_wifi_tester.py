import requests

ESP32_IP = "http://192.168.47.170"  # Update if your IP changes

def send_servo_angles(s1=None, s2=None):
    params = []
    if s1 is not None:
        params.append(f"angle1={s1}")
    if s2 is not None:
        params.append(f"angle2={s2}")
    query = "&".join(params)
    url = f"{ESP32_IP}/servo?{query}"
    try:
        response = requests.get(url)
        print("ESP32 response:", response.text)
    except requests.exceptions.RequestException as e:
        print("Failed to connect to ESP32:", e)

# Example
send_servo_angles(s1=82, s2=115)