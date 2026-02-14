
import sys
import os

# Add backend to path so we can import ml modules
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

try:
    from ml.inference_engine import InferenceEngine
    print("✓ Successfully imported InferenceEngine")
except ImportError as e:
    print(f"✗ Failed to import InferenceEngine: {e}")
    sys.exit(1)

def main():
    print("Initializing Inference Engine...")
    engine = InferenceEngine()
    
    test_questions = [
        "What is linear regression?",
        "How do I choose the right k in K-means clustering?",
        "Can you explain gradient descent briefly?"
    ]
    
    for question in test_questions:
        print(f"\nUser: {question}")
        print("Tutor: ", end="", flush=True)
        response = engine.generate_response(question)
        print(response)

if __name__ == "__main__":
    main()
