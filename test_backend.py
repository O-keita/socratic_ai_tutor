import requests
import json

def test_chat():
    url = "http://127.0.0.1:8000/chat"
    payload = {
        "message": "What is a neural network?",
        "history": [],
        "difficulty": "beginner"
    }
    headers = {
        "Content-Type": "application/json"
    }

    print(f"Sending request to {url}...")
    try:
        response = requests.post(url, json=payload, headers=headers, timeout=60)
        
        if response.status_code == 200:
            result = response.json()
            print("\n✅ Success!")
            print(f"Tutor Response: {result.get('response')}")
        else:
            print(f"\n❌ Error: {response.status_code}")
            print(response.text)
            
    except requests.exceptions.ConnectionError:
        print("\n❌ Connection Error: Is the backend server running?")
    except Exception as e:
        print(f"\n❌ Unexpected Error: {e}")

if __name__ == "__main__":
    test_chat()
