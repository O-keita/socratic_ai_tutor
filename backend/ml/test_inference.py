import sys
import os

# Add backend to path so we can import ml modules
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

try:
    from ml.inference_engine import inference_engine
    print("✓ Successfully imported InferenceEngine")
except ImportError as e:
    print(f"✗ Failed to import InferenceEngine: {e}")
    sys.exit(1)

def main():
    print("\n--- Socratic AI Tutor Interactive Test ---")
    print("Type 'exit' or 'quit' to stop.\n")
    
    history = []
    
    while True:
        try:
            user_input = input("User: ")
            if user_input.lower() in ['exit', 'quit']:
                break
                
            if not user_input.strip():
                continue
                
            print("Tutor: ", end="", flush=True)
            
            # Generate response using history
            response = inference_engine.generate_response(user_input, history=history)
            print(response)
            
            # Update history for next turn
            history.append({"role": "user", "content": user_input})
            history.append({"role": "assistant", "content": response})
            
            # Keep history manageable if it gets too long
            if len(history) > 10:
                history = history[-10:]
                
        except KeyboardInterrupt:
            print("\nExiting...")
            break
        except Exception as e:
            print(f"\nError: {e}")

if __name__ == "__main__":
    main()
