#!/usr/bin/env python3
"""
LWE Attack Fast Demo Script

This script demonstrates a machine learning attack on LWE using a subset of models
for faster execution. It shows the attack methodology without loading all 512 models.
"""

import numpy as np
import os
import sys
import json
import pickle
from pathlib import Path
import time

# Force TensorFlow to use CPU only
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf
from tensorflow import keras

# LWE Parameters
n = 512
q = 12289
p = 4
delta = q // p

class LWEAttackFastDemo:
    """Fast LWE attack demo using subset of models"""
    
    def __init__(self, model_dir="lwe_chunked_models", num_models_to_load=20):
        self.n = n
        self.q = q
        self.model_dir = Path(model_dir)
        self.num_models_to_load = num_models_to_load
        self.models = {}  # Dictionary to store loaded models
        self.metadata = None
        
    def load_sample_models(self):
        """Load a subset of models for demonstration"""
        print(f"=== Loading {self.num_models_to_load} Sample Models ===")
        
        # Load metadata
        metadata_path = self.model_dir / "metadata.json"
        if not metadata_path.exists():
            raise FileNotFoundError(f"Metadata file not found: {metadata_path}")
            
        with open(metadata_path, 'r') as f:
            self.metadata = json.load(f)
            
        print(f"Found metadata for {self.metadata['num_models']} total models")
        print(f"Loading first {self.num_models_to_load} models for demonstration...")
        
        # Load first N models
        loaded_count = 0
        for i in range(min(self.num_models_to_load, self.n)):
            model_path = self.model_dir / f"bit_model_{i:03d}.h5"
            if model_path.exists():
                try:
                    print(f"Loading model {i}...", end='\r')
                    self.models[i] = keras.models.load_model(str(model_path))
                    loaded_count += 1
                except Exception as e:
                    print(f"Warning: Could not load model {i}: {e}")
                    
        print(f"Successfully loaded {loaded_count} models")
        return loaded_count
        
    def load_original_secret(self):
        """Load the original secret key"""
        try:
            data_path = self.model_dir / "training_data.pkl"
            with open(data_path, 'rb') as f:
                data = pickle.load(f)
            return data['secret']
        except FileNotFoundError:
            print("Warning: Original secret key not found")
            return None
            
    def encrypt_message(self, message, secret_key, noise_std=3.2):
        """Encrypt a message using LWE"""
        a = np.random.randint(0, self.q, self.n)
        noise = int(np.round(np.random.normal(0, noise_std)))
        dot_product = np.dot(a, secret_key) % self.q
        b = (dot_product + delta * message + noise) % self.q
        return a, b
        
    def predict_secret_bits(self, sample_a):
        """Predict secret bits using available models"""
        predicted_bits = {}
        confidences = {}
        
        input_data = sample_a.reshape(1, -1)
        
        print(f"Predicting secret bits using {len(self.models)} trained models...")
        
        for i, model in self.models.items():
            try:
                prob = model.predict(input_data, verbose=0)[0][0]
                bit_prediction = 1 if prob > 0.5 else 0
                confidence = max(prob, 1 - prob)
                
                predicted_bits[i] = bit_prediction
                confidences[i] = confidence
                
            except Exception as e:
                print(f"Warning: Error predicting with model {i}: {e}")
                
        return predicted_bits, confidences
        
    def demonstrate_partial_attack(self):
        """Demonstrate attack with partial secret knowledge"""
        print("=== LWE Attack Demo (Partial Secret Recovery) ===\n")
        
        # Load sample models
        loaded_count = self.load_sample_models()
        if loaded_count == 0:
            print("Error: No models could be loaded!")
            return
            
        # Load original secret
        original_secret = self.load_original_secret()
        if original_secret is None:
            print("Error: Could not load original secret for demonstration")
            return
            
        print(f"\nOriginal secret (first 20 bits): {original_secret[:20]}")
        
        # Generate test message
        test_message = 2
        print(f"Testing with message: {test_message}")
        
        # Encrypt message
        a, b = self.encrypt_message(test_message, original_secret)
        print(f"Encrypted: a=(vector), b={b}")
        
        # Predict available secret bits
        predicted_bits, confidences = self.predict_secret_bits(a)
        
        print(f"\n=== Detailed Secret Key Comparison ===")
        print("Bit | Original | Predicted | Confidence | Match")
        print("-" * 45)
        
        correct_predictions = 0
        total_predictions = len(predicted_bits)
        
        for i in sorted(predicted_bits.keys()):
            original_bit = original_secret[i]
            predicted_bit = predicted_bits[i]
            confidence = confidences[i]
            is_correct = original_bit == predicted_bit
            
            if is_correct:
                correct_predictions += 1
                
            status = "✓" if is_correct else "✗"
            print(f"{i:2d}  |    {original_bit}     |     {predicted_bit}     |   {confidence:.3f}    |  {status}")
            
        accuracy = correct_predictions / total_predictions
        print(f"\nPartial secret recovery accuracy: {correct_predictions}/{total_predictions} ({accuracy:.1%})")
        
        # Show side-by-side comparison
        print(f"\n=== Side-by-Side Secret Key Comparison ===")
        max_bits = max(predicted_bits.keys()) + 1
        print(f"Original  (bits 0-{max_bits-1}): {' '.join(str(original_secret[i]) for i in range(max_bits))}")
        
        predicted_display = []
        for i in range(max_bits):
            if i in predicted_bits:
                predicted_display.append(str(predicted_bits[i]))
            else:
                predicted_display.append('?')
        print(f"Predicted (bits 0-{max_bits-1}): {' '.join(predicted_display)}")
        
        # Show match status
        match_display = []
        for i in range(max_bits):
            if i in predicted_bits:
                match_display.append('✓' if original_secret[i] == predicted_bits[i] else '✗')
            else:
                match_display.append('?')
        print(f"Match     (bits 0-{max_bits-1}): {' '.join(match_display)}")
        
        # Demonstrate attack limitation
        print(f"\n=== Attack Analysis ===")
        print(f"Models loaded: {len(self.models)}/{self.n} ({len(self.models)/self.n:.1%})")
        print(f"Average model confidence: {np.mean(list(confidences.values())):.3f}")
        
        if accuracy > 0.7:
            print("✓ High accuracy - Attack appears successful on loaded bits")
        else:
            print("✗ Low accuracy - Attack may need more training or different approach")
            
        # Show what full attack would look like
        print(f"\n=== Full Attack Simulation ===")
        print("With all 512 models loaded, the attack would:")
        print("1. Predict all 512 secret bits")
        print("2. Use complete predicted secret to decrypt messages")
        print("3. Achieve near-perfect decryption if models are well-trained")
        
        # Simulate full secret (mix of predicted and random)
        full_predicted_secret = np.random.randint(0, 2, self.n)
        for i, bit in predicted_bits.items():
            full_predicted_secret[i] = bit
            
        # Try to decrypt with partial secret
        dot_product = np.dot(a, full_predicted_secret) % self.q
        noisy_message = (b - dot_product) % self.q
        decrypted_message = round(noisy_message / delta) % p
        
        print(f"\nDecryption with partial secret:")
        print(f"Original message: {test_message}")
        print(f"Decrypted message: {decrypted_message}")
        
        if decrypted_message == test_message:
            print("✓ Decryption successful!")
        else:
            print("✗ Decryption failed (expected with partial secret)")
            
    def interactive_bit_prediction(self):
        """Interactive demo for bit prediction"""
        print("\n=== Interactive Bit Prediction Demo ===")
        print("Enter custom vectors to see bit predictions, or 'q' to quit")
        
        original_secret = self.load_original_secret()
        
        while True:
            try:
                user_input = input(f"\nEnter 'test' for random vector, or 'q' to quit: ").strip()
                
                if user_input.lower() == 'q':
                    break
                elif user_input.lower() == 'test':
                    # Generate random test vector
                    test_vector = np.random.randint(0, self.q, self.n)
                    print(f"Generated random test vector (first 10 elements): {test_vector[:10]}")
                    
                    # Predict bits
                    predicted_bits, confidences = self.predict_secret_bits(test_vector)
                    
                    print("Predicted bits:")
                    for i in sorted(predicted_bits.keys())[:10]:  # Show first 10
                        bit = predicted_bits[i]
                        conf = confidences[i]
                        original = original_secret[i] if original_secret is not None else '?'
                        print(f"  Bit {i}: {bit} (confidence: {conf:.3f}, original: {original})")
                        
                else:
                    print("Invalid input. Enter 'test' or 'q'.")
                    
            except (ValueError, KeyboardInterrupt):
                print("Invalid input or interrupted")
                break
                
        print("Interactive demo finished")

def main():
    """Main function"""
    print("LWE Attack Fast Demo")
    print("=" * 50)
    
    # Check if models exist
    model_dir = Path("lwe_chunked_models")
    if not model_dir.exists():
        print(f"Error: Model directory '{model_dir}' not found!")
        return 1
        
    # Initialize demo
    demo = LWEAttackFastDemo(model_dir, num_models_to_load=20)
    
    try:
        # Run demonstration
        demo.demonstrate_partial_attack()
        
        # Ask for interactive demo
        interactive = input("\nWould you like to try interactive bit prediction? (y/n): ").lower().strip()
        if interactive == 'y':
            demo.interactive_bit_prediction()
            
    except Exception as e:
        print(f"Error running demo: {e}")
        import traceback
        traceback.print_exc()
        return 1
        
    print("\nDemo completed!")
    print("\nNote: This demo used only a subset of models for speed.")
    print("The full attack would load all 512 models for complete secret recovery.")
    return 0

if __name__ == "__main__":
    sys.exit(main())