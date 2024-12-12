import requests

webhook_url = "https://discord.com/api/webhooks/1316764358222549024/V5fcMESK3rIGunDg4tMts3jJ6dNe8hvz6dUUJqk-gCI3I1irwMoG-o10kqxR_xT9Qgyk"

payload = {
    "content": "Hello, this is a test message from the webhook!",
    "username": "Test Bot"
}

headers = {
    "Content-Type": "application/json"
}

response = requests.post(webhook_url, json=payload, headers=headers)

if response.status_code == 204:
    print("Message sent successfully!")
else:
    print(f"Failed to send message. HTTP {response.status_code}: {response.text}")

