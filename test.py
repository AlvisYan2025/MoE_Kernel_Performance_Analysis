#test script to check model
import requests
import json
import time

def test_inference(port=8000):
    url = f"http://localhost:{port}/v1/completions"
    
    data = {
        "prompt": "Explain MoE models in simple terms:",
        "max_tokens": 100,
        "temperature": 0.7
    }
    print("Sending test request...")
    start_time = time.time()
    response = requests.post(url, json=data)
    end_time = time.time()
    
    if response.status_code == 200:
        result = response.json()
        print(f"\n{'='*50}")
        print(f"Response received in {end_time - start_time:.2f}s")
        print(f"{'='*50}")
        print(f"\nPrompt: {data['prompt']}")
        print(f"\nCompletion: {result['choices'][0]['text']}")
        print(f"\n{'='*50}")
        print(f"Usage: {result['usage']}")
    else:
        print(f"Error: {response.status_code}")
        print(response.text)

if __name__ == "__main__":
    test_inference()