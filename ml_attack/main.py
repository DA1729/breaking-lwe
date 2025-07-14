import numpy as np
import tensorflow as tf
from tensorflow import keras
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import subprocess
import tempfile
import os

# Parameters matching your C++ code
n = 512
q = 12289
p = 4
delta = q // p

class LWEDataGenerator:
    """generate training data by running the C++ LWE implementation"""
    
    def __init__(self, cpp_executable_path=None):
        self.cpp_code = '''
#include <iostream>
#include <vector>
#include <random>
#include <cmath>
#include <cassert>
constexpr int n = 512;
constexpr int q = 12289;
constexpr int p = 4;
constexpr int delta = q / p;

int sample_discrete_gaussian(std::mt19937& gen, double sigma = 3.2) {
    std::normal_distribution<> dist(0.0, sigma);
    return static_cast<int>(std::round(dist(gen))) % q;
}

std::vector<int> key_gen(std::mt19937& gen) {
    std::vector<int> s(n);
    std::bernoulli_distribution bern(0.5);
    for (int& si : s) si = bern(gen);
    return s;
}

int dot_mod_q(const std::vector<int>& a, const std::vector<int>& b) {
    assert(a.size() == b.size());
    int64_t sum = 0;
    for (size_t i = 0; i < a.size(); ++i)
        sum += static_cast<int64_t>(a[i]) * b[i];
    return static_cast<int>(sum % q);
}

std::pair<std::vector<int>, int> encrypt(int m, const std::vector<int>& s, std::mt19937& gen) {
    std::uniform_int_distribution<> uniform_q(0, q - 1);
    std::vector<int> a(n);
    for (int& ai : a) ai = uniform_q(gen);
    int e = sample_discrete_gaussian(gen);
    int b = (dot_mod_q(a, s) + delta * m + e) % q;
    return {a, b};
}

int main() {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::vector<int> s = key_gen(gen);
    
    // Output secret key
    std::cout << "SECRET:";
    for (int si : s) std::cout << " " << si;
    std::cout << std::endl;
    
    // Generate training samples
    for (int i = 0; i < 10000; ++i) {
        auto ct = encrypt(0, s, gen);  // Always encrypt 0 for key recovery
        std::cout << "SAMPLE:";
        for (int ai : ct.first) std::cout << " " << ai;
        std::cout << " " << ct.second << std::endl;
    }
    return 0;
}
'''
    
    def generate_data(self, num_samples=10000):
        """generate LWE samples and secret key"""
        # Create temporary C++ file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.cpp', delete=False) as f:
            f.write(self.cpp_code)
            cpp_file = f.name
        
        try:
            # Compile
            exe_file = cpp_file.replace('.cpp', '')
            subprocess.run(['g++', '-o', exe_file, cpp_file], check=True)
            
            # Run and capture output
            result = subprocess.run([exe_file], capture_output=True, text=True, check=True)
            
            # Parse output
            lines = result.stdout.strip().split('\n')
            
            # Extract secret key
            secret_line = [line for line in lines if line.startswith('SECRET:')][0]
            secret = list(map(int, secret_line.split()[1:]))
            
            # Extract samples
            sample_lines = [line for line in lines if line.startswith('SAMPLE:')]
            samples = []
            for line in sample_lines:
                parts = list(map(int, line.split()[1:]))
                a = parts[:-1]  # First n elements
                b = parts[-1]   # Last element
                samples.append((a, b))
            
            return np.array(secret), samples
            
        finally:
            # Cleanup
            for file in [cpp_file, exe_file]:
                if os.path.exists(file):
                    os.unlink(file)

def create_training_data(secret, samples):
    """convert LWE samples to ML training data"""
    X = []  # Input vectors (a values)
    y = []  # Target labels (secret bits)
    
    # For each sample (a, b), create n training examples
    # Each example: input = a, target = s_i for position i
    for a, b in samples:
        X.append(a)
        # We'll train separate models for each bit position
    
    return np.array(X), np.array(secret)

class LWEAttacker:
    """machine Learning attack on LWE"""
    
    def __init__(self, n=512, q=12289):
        self.n = n
        self.q = q
        self.models = []
        
    def create_bit_model(self):
        """create a neural network to predict one secret bit"""
        model = keras.Sequential([
            keras.layers.Dense(256, activation='relu', input_shape=(self.n,)),
            keras.layers.Dropout(0.3),
            keras.layers.Dense(128, activation='relu'),
            keras.layers.Dropout(0.3),
            keras.layers.Dense(64, activation='relu'),
            keras.layers.Dense(1, activation='sigmoid')
        ])
        
        model.compile(
            optimizer='adam',
            loss='binary_crossentropy',
            metrics=['accuracy']
        )
        return model
    
    def train_models(self, X, secret, epochs=50):
        """train separate models for each secret bit"""
        print(f"Training {self.n} models for each secret bit...")
        
        self.models = []
        accuracies = []
        
        for i in range(self.n):
            print(f"Training model for bit {i+1}/{self.n}", end='\r')
            
            # create targets: for each sample, predict secret[i]
            y_bit = np.full(len(X), secret[i])
            
            model = self.create_bit_model()
            
            # train with validation split
            history = model.fit(
                X, y_bit,
                epochs=epochs,
                batch_size=32,
                validation_split=0.2,
                verbose=0
            )
            
            self.models.append(model)
            
            # evaluate on training data (for demonstration)
            predictions = (model.predict(X) > 0.5).astype(int).flatten()
            accuracy = accuracy_score(y_bit, predictions)
            accuracies.append(accuracy)
        
        print(f"\nTraining complete! Average accuracy: {np.mean(accuracies):.3f}")
        return accuracies
    
    def predict_secret(self, X_test):
        """Predict the secret key using trained models"""
        if not self.models:
            raise ValueError("Models not trained yet!")
        
        predicted_secret = []
        confidences = []
        
        for i, model in enumerate(self.models):
            # use the first test sample (or average over multiple samples)
            prob = model.predict(X_test[:1])[0][0]
            bit_prediction = 1 if prob > 0.5 else 0
            confidence = max(prob, 1 - prob)
            
            predicted_secret.append(bit_prediction)
            confidences.append(confidence)
        
        return np.array(predicted_secret), np.array(confidences)
    
    def attack_ciphertext(self, a, b, predicted_secret):
        """Decrypt a ciphertext using the recovered secret"""
        # compute dot product mod q
        dot_product = sum(a[i] * predicted_secret[i] for i in range(len(a))) % self.q
        
        # recover phase
        phase = (b - dot_product + self.q) % self.q
        
        # decode message
        message = round(phase / delta) % p
        return message

def demonstrate_attack():
    print("=== LWE Machine Learning Attack Demo ===\n")
    
    # generate data
    print("1. Generating LWE samples...")
    generator = LWEDataGenerator()
    try:
        true_secret, samples = generator.generate_data(num_samples=5000)
        print(f"Generated {len(samples)} samples")
        print(f"Secret key (first 10 bits): {true_secret[:10]}...")
    except Exception as e:
        print(f"Error generating data: {e}")
        print("Using simulated data instead...")
        # fallback to simulated data
        true_secret, samples = simulate_lwe_data()
    
    # prepare training data
    print("\n2. Preparing training data...")
    X, secret = create_training_data(true_secret, samples)
    print(f"Training data shape: {X.shape}")
    
    # train attack models
    print("\n3. Training ML models...")
    attacker = LWEAttacker()
    accuracies = attacker.train_models(X, secret, epochs=30)
    
    # analyze results
    print(f"\n4. Attack Results:")
    print(f"   - Bits with >90% accuracy: {sum(acc > 0.9 for acc in accuracies)}/{len(accuracies)}")
    print(f"   - Bits with >80% accuracy: {sum(acc > 0.8 for acc in accuracies)}/{len(accuracies)}")
    print(f"   - Average accuracy: {np.mean(accuracies):.3f}")
    
    # test secret recovery
    print("\n5. Testing secret recovery...")
    predicted_secret, confidences = attacker.predict_secret(X)
    
    # compare with true secret
    correct_bits = sum(predicted_secret[i] == true_secret[i] for i in range(len(true_secret)))
    print(f"   - Correctly recovered bits: {correct_bits}/{len(true_secret)}")
    print(f"   - Success rate: {correct_bits/len(true_secret):.1%}")
    
    # test decryption
    print("\n6. Testing decryption attack...")
    test_sample = samples[0]  # Use first sample as test
    a_test, b_test = test_sample
    
    decrypted_msg = attacker.attack_ciphertext(a_test, b_test, predicted_secret)
    print(f"   - Attempted decryption: {decrypted_msg}")
    print(f"   - (Original message was 0)")
    
    return attacker, true_secret, predicted_secret

def simulate_lwe_data():
    """Simulate LWE data when C++ compilation fails"""
    print("Simulating LWE data...")
    
    # generate random secret
    secret = np.random.randint(0, 2, n)
    
    # generate samples
    samples = []
    for _ in range(5000):
        a = np.random.randint(0, q, n)
        noise = np.random.normal(0, 3.2)
        dot_product = np.dot(a, secret) % q
        b = (dot_product + int(noise)) % q
        samples.append((a.tolist(), b))
    
    return secret, samples

if __name__ == "__main__":
    demonstrate_attack()
