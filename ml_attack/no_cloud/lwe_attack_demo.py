#!/usr/bin/env python3
"""
LWE Attack Demonstration Script

This script demonstrates a machine learning attack on the Learning With Errors (LWE) problem
using pre-trained models. It loads trained neural networks that predict each bit of the secret
key and uses them to decrypt LWE ciphertexts.

Note: This is for educational/research purposes to understand LWE security.
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
from sklearn.metrics import accuracy_score

# LWE Parameters (matching the trained models)
n = 512
q = 12289
p = 4
delta = q // p

class LWEAttackDemo:
    """Demonstrates LWE attack using pre-trained models"""
    
    def __init__(self, model_dir="lwe_chunked_models"):
        self.n = n
        self.q = q
        self.model_dir = Path(model_dir)
        self.models = [None] * n
        self.metadata = None
        
    def load_models(self):
        """Load all trained models"""
        print("=== Loading Pre-trained Models ===")
        
        # Load metadata
        metadata_path = self.model_dir / "metadata.json"
        if not metadata_path.exists():
            raise FileNotFoundError(f"Metadata file not found: {metadata_path}")
            
        with open(metadata_path, 'r') as f:
            self.metadata = json.load(f)
            
        print(f"Found metadata for {self.metadata['num_models']} models")
        print(f"Valid models: {self.metadata['valid_models']}")
        
        # Load each model
        loaded_count = 0
        failed_models = []
        
        for i in range(self.n):
            model_path = self.model_dir / f"bit_model_{i:03d}.h5"
            if model_path.exists():
                try:
                    self.models[i] = keras.models.load_model(str(model_path))
                    loaded_count += 1
                    print(f"Loading model {i}...", end='\r')
                except Exception as e:
                    print(f"Warning: Could not load model {i}: {e}")
                    failed_models.append(i)
            else:
                failed_models.append(i)
                
        print(f"Successfully loaded {loaded_count}/{self.n} models")
        if failed_models:
            print(f"Failed to load {len(failed_models)} models")
            
        return loaded_count
        
    def load_original_secret(self):
        """Load the original secret key used for training"""
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
        
    def predict_secret_key(self, sample_a):
        """Predict the secret key using trained models"""
        predicted_secret = []
        confidences = []
        
        # Reshape for neural network input
        input_data = sample_a.reshape(1, -1)
        
        for i, model in enumerate(self.models):
            if model is not None:
                try:
                    prob = model.predict(input_data, verbose=0)[0][0]
                    bit_prediction = 1 if prob > 0.5 else 0
                    confidence = max(prob, 1 - prob)
                except Exception as e:
                    print(f"Warning: Error predicting with model {i}: {e}")
                    bit_prediction = 0
                    confidence = 0.5
            else:
                # If model not available, random guess
                bit_prediction = np.random.randint(0, 2)
                confidence = 0.5
                
            predicted_secret.append(bit_prediction)
            confidences.append(confidence)
            
        return np.array(predicted_secret), np.array(confidences)
        
    def decrypt_ciphertext(self, a, b, predicted_secret):
        """Decrypt a ciphertext using predicted secret key"""
        # Compute dot product mod q
        dot_product = np.dot(a, predicted_secret) % self.q
        
        # Recover the noisy message
        noisy_message = (b - dot_product) % self.q
        
        # Decode the message
        message = round(noisy_message / delta) % p
        return message
        
    def display_secret_comparison(self, original_secret, predicted_secret, confidences, num_display=50):
        """Display detailed comparison between original and predicted secret bits"""
        print(f"\n=== Secret Key Comparison (First {num_display} bits) ===")
        print("Bit  | Original | Predicted | Confidence | Match")
        print("-" * 48)
        
        correct_count = 0
        total_count = min(num_display, len(original_secret))
        
        for i in range(total_count):
            original_bit = original_secret[i]
            predicted_bit = predicted_secret[i]
            confidence = confidences[i]
            match = "✓" if original_bit == predicted_bit else "✗"
            
            if original_bit == predicted_bit:
                correct_count += 1
                
            print(f"{i:3d}  |    {original_bit}     |     {predicted_bit}     |   {confidence:.3f}    |  {match}")
        
        accuracy = correct_count / total_count
        print(f"\nAccuracy: {correct_count}/{total_count} ({accuracy:.1%})")
        
        # Show side-by-side comparison
        print(f"\n=== Side-by-Side Secret Key Comparison ===")
        print(f"Original  (bits 0-{num_display-1}): {' '.join(str(original_secret[i]) for i in range(num_display))}")
        print(f"Predicted (bits 0-{num_display-1}): {' '.join(str(predicted_secret[i]) for i in range(num_display))}")
        
        # Show match status
        match_display = ['✓' if original_secret[i] == predicted_secret[i] else '✗' for i in range(num_display)]
        print(f"Match     (bits 0-{num_display-1}): {' '.join(match_display)}")
        
        return correct_count, total_count
        
    def run_attack_demo(self):
        """Run the complete attack demonstration"""
        print("=== LWE Machine Learning Attack Demo ===\n")
        
        # Load models
        loaded_count = self.load_models()
        if loaded_count == 0:
            print("Error: No models could be loaded!")
            return
            
        # Load original secret (if available)
        original_secret = self.load_original_secret()
        if original_secret is None:
            print("Error: Could not load original secret for comparison")
            return
            
        # Generate test messages
        test_messages = [0, 1, 2, 3]
        
        print(f"\n=== Attack Demonstration ===")
        print(f"Testing decryption of messages: {test_messages}")
        
        # For each test message, encrypt it and try to decrypt
        successful_decryptions = 0
        
        for message in test_messages:
            print(f"\n--- Testing Message: {message} ---")
            
            # Encrypt the message
            a, b = self.encrypt_message(message, original_secret)
            print(f"Encrypted: a=(vector of length {len(a)}), b={b}")
            
            # Predict the secret key using our trained models
            start_time = time.time()
            predicted_secret, confidences = self.predict_secret_key(a)
            prediction_time = time.time() - start_time
            
            # Decrypt using predicted secret
            decrypted_message = self.decrypt_ciphertext(a, b, predicted_secret)
            
            # Check if decryption was successful
            success = decrypted_message == message
            if success:
                successful_decryptions += 1
                
            print(f"Predicted secret key in {prediction_time:.3f}s")
            print(f"Average model confidence: {np.mean(confidences):.3f}")
            print(f"Decrypted message: {decrypted_message}")
            print(f"Decryption {'SUCCESS' if success else 'FAILED'}")
            
            # Show detailed secret comparison
            correct_bits, total_bits = self.display_secret_comparison(
                original_secret, predicted_secret, confidences, num_display=30
            )
            
            print(f"Secret recovery accuracy: {correct_bits}/{total_bits} ({correct_bits/total_bits:.1%})")
                
        print(f"\n=== Final Attack Results ===")
        print(f"Successfully decrypted {successful_decryptions}/{len(test_messages)} messages")
        print(f"Overall success rate: {successful_decryptions/len(test_messages):.1%}")
        
        # Additional analysis
        if self.metadata:
            avg_model_accuracy = np.mean(self.metadata['accuracies'])
            print(f"Average model training accuracy: {avg_model_accuracy:.3f}")
            
        return successful_decryptions / len(test_messages)

def main():
    """Main function"""
    print("LWE Attack Demonstration")
    print("=" * 50)
    
    # Check if models exist
    model_dir = Path("lwe_chunked_models")
    if not model_dir.exists():
        print(f"Error: Model directory '{model_dir}' not found!")
        print("Please ensure you have trained models in this directory.")
        return 1
        
    # Initialize attack demo
    attack_demo = LWEAttackDemo(model_dir)
    
    try:
        # Run the main demonstration
        success_rate = attack_demo.run_attack_demo()
        
    except Exception as e:
        print(f"Error running attack demo: {e}")
        import traceback
        traceback.print_exc()
        return 1
        
    print("\nDemo completed!")
    return 0

if __name__ == "__main__":
    sys.exit(main())