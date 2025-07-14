import numpy as np
import os
# Force TensorFlow to use CPU only to avoid CUDA issues
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Suppress TensorFlow warnings

import tensorflow as tf
from tensorflow import keras
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import subprocess
import tempfile
import json
import pickle
from pathlib import Path

# Parameters matching your C++ code
n = 512
q = 12289
p = 4
delta = q // p

class LWEDataGenerator:
    """Generate training data by running the C++ LWE implementation"""

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
        """Generate LWE samples and secret key"""
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
    """Convert LWE samples to ML training data"""
    X = []  # Input vectors (a values)
    y = []  # Target labels (secret bits)

    # For each sample (a, b), create n training examples
    # Each example: input = a, target = s_i for position i
    for a, b in samples:
        X.append(a)
        # We'll train separate models for each bit position

    return np.array(X), np.array(secret)

class LWEAttacker:
    """Machine Learning attack on LWE with chunked training support"""

    def __init__(self, n=512, q=12289, model_dir="lwe_models"):
        self.n = n
        self.q = q
        self.model_dir = Path(model_dir)
        self.models = [None] * n  # Initialize with None placeholders
        self.is_trained = False

        # Create model directory if it doesn't exist
        self.model_dir.mkdir(exist_ok=True)

    def create_bit_model(self):
        """Create a neural network to predict one secret bit"""
        # Use Input layer to avoid warnings and ensure CPU compatibility
        model = keras.Sequential([
            keras.layers.Input(shape=(self.n,)),
            keras.layers.Dense(256, activation='relu'),
            keras.layers.Dropout(0.3),
            keras.layers.Dense(128, activation='relu'),
            keras.layers.Dropout(0.3),
            keras.layers.Dense(64, activation='relu'),
            keras.layers.Dense(1, activation='sigmoid')
        ])

        # Use CPU-friendly optimizer settings
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.001),
            loss='binary_crossentropy',
            metrics=['accuracy']
        )
        return model

    def save_training_data(self, X, secret):
        """Save training data for consistent use across chunks"""
        data_path = self.model_dir / "training_data.pkl"
        with open(data_path, 'wb') as f:
            pickle.dump({'X': X, 'secret': secret}, f)
        print(f"Training data saved to {data_path}")

    def load_training_data(self):
        """Load saved training data"""
        data_path = self.model_dir / "training_data.pkl"
        if not data_path.exists():
            raise FileNotFoundError(f"Training data not found at {data_path}")

        with open(data_path, 'rb') as f:
            data = pickle.load(f)
        return data['X'], data['secret']

    def get_training_progress(self):
        """Get current training progress"""
        metadata_path = self.model_dir / "training_progress.json"
        if not metadata_path.exists():
            return {
                'trained_models': [],
                'accuracies': [0.0] * self.n,
                'current_chunk': 0,
                'total_chunks': 0
            }

        with open(metadata_path, 'r') as f:
            return json.load(f)

    def save_training_progress(self, trained_models, accuracies, current_chunk, total_chunks):
        """Save training progress"""
        progress = {
            'trained_models': trained_models,
            'accuracies': accuracies,
            'current_chunk': current_chunk,
            'total_chunks': total_chunks,
            'n': self.n,
            'q': self.q
        }

        with open(self.model_dir / "training_progress.json", 'w') as f:
            json.dump(progress, f, indent=2)

    def save_model_chunk(self, model_indices, chunk_accuracies):
        """Save models for a specific chunk"""
        for i, idx in enumerate(model_indices):
            if self.models[idx] is not None:
                model_path = self.model_dir / f"bit_model_{idx:03d}.h5"
                try:
                    self.models[idx].save(str(model_path))
                except Exception as e:
                    print(f"Warning: Could not save model {idx}: {e}")

    def load_existing_models(self):
        """Load any existing trained models"""
        progress = self.get_training_progress()
        trained_models = progress['trained_models']

        loaded_count = 0
        for idx in trained_models:
            model_path = self.model_dir / f"bit_model_{idx:03d}.h5"
            if model_path.exists():
                try:
                    self.models[idx] = keras.models.load_model(str(model_path))
                    loaded_count += 1
                except Exception as e:
                    print(f"Warning: Could not load model {idx}: {e}")
                    self.models[idx] = None

        return loaded_count, progress['accuracies']

    def train_models_chunk(self, X, secret, start_idx, end_idx, epochs=50, chunk_num=1, total_chunks=1):
        """Train models for a specific chunk (start_idx to end_idx)"""
        print(f"\n=== Training Chunk {chunk_num}/{total_chunks} ===")
        print(f"Training models {start_idx} to {end_idx-1} ({end_idx-start_idx} models)")
        print("Note: Training on CPU to avoid CUDA issues. This may take longer.")

        # Load existing progress
        progress = self.get_training_progress()
        trained_models = progress['trained_models']
        accuracies = progress['accuracies']

        chunk_indices = list(range(start_idx, min(end_idx, self.n)))

        for i, idx in enumerate(chunk_indices):
            if idx in trained_models:
                print(f"Model {idx} already trained, skipping...")
                continue

            progress_pct = (i + 1) / len(chunk_indices) * 100
            print(f"Training model {idx} ({i+1}/{len(chunk_indices)}) - {progress_pct:.1f}% of chunk", end='\r')

            # Create targets: for each sample, predict secret[idx]
            y_bit = np.full(len(X), secret[idx])

            model = self.create_bit_model()

            # Train with validation split and reduced verbosity
            try:
                history = model.fit(
                    X, y_bit,
                    epochs=epochs,
                    batch_size=32,
                    validation_split=0.2,
                    verbose=0,
                    shuffle=True
                )

                self.models[idx] = model

                # Evaluate on training data
                predictions = (model.predict(X, verbose=0) > 0.5).astype(int).flatten()
                accuracy = accuracy_score(y_bit, predictions)
                accuracies[idx] = accuracy

                # Add to trained models list
                trained_models.append(idx)

                # Save progress periodically (every 10 models)
                if len(trained_models) % 10 == 0:
                    self.save_training_progress(trained_models, accuracies, chunk_num, total_chunks)

            except Exception as e:
                print(f"\nError training model {idx}: {e}")
                print("Continuing with next model...")
                accuracies[idx] = 0.0
                continue

        # Save models in this chunk
        self.save_model_chunk(chunk_indices, accuracies)

        # Save final progress for this chunk
        self.save_training_progress(trained_models, accuracies, chunk_num, total_chunks)

        print(f"\nChunk {chunk_num} complete!")
        chunk_accuracies = [accuracies[i] for i in chunk_indices if i in trained_models]
        if chunk_accuracies:
            print(f"Average accuracy for this chunk: {np.mean(chunk_accuracies):.3f}")

        return accuracies

    def train_models_chunked(self, X, secret, chunk_size=100, epochs=50, save_data=True):
        """Train all models in chunks"""
        print(f"=== Chunked Training: {self.n} models in chunks of {chunk_size} ===")

        # Save training data for consistency across chunks
        if save_data:
            self.save_training_data(X, secret)

        # Calculate chunks
        total_chunks = (self.n + chunk_size - 1) // chunk_size

        # Load any existing models
        loaded_count, accuracies = self.load_existing_models()
        if loaded_count > 0:
            print(f"Loaded {loaded_count} existing models")

        # Train each chunk
        for chunk_num in range(1, total_chunks + 1):
            start_idx = (chunk_num - 1) * chunk_size
            end_idx = min(start_idx + chunk_size, self.n)

            # Check if chunk is already complete
            progress = self.get_training_progress()
            trained_in_chunk = [i for i in range(start_idx, end_idx) if i in progress['trained_models']]

            if len(trained_in_chunk) == (end_idx - start_idx):
                print(f"Chunk {chunk_num} already complete, skipping...")
                continue

            print(f"\nStarting chunk {chunk_num}/{total_chunks}")
            print(f"You can stop after this chunk and resume later if needed.")

            accuracies = self.train_models_chunk(X, secret, start_idx, end_idx, epochs, chunk_num, total_chunks)

            # Ask user if they want to continue (except for last chunk)
            if chunk_num < total_chunks:
                print(f"\nChunk {chunk_num} completed successfully!")
                print(f"Progress: {len(progress['trained_models']) + (end_idx - start_idx)}/{self.n} models trained")

                continue_training = input("Continue to next chunk? (y/n): ").lower().strip()
                if continue_training != 'y':
                    print("Training paused. Run again to continue from where you left off.")
                    break

        # Final summary
        progress = self.get_training_progress()
        total_trained = len(progress['trained_models'])

        print(f"\n=== Training Summary ===")
        print(f"Total models trained: {total_trained}/{self.n}")
        print(f"Completion: {total_trained/self.n:.1%}")

        if total_trained == self.n:
            print("All models trained successfully!")
            avg_accuracy = np.mean([acc for acc in progress['accuracies'] if acc > 0])
            print(f"Average accuracy: {avg_accuracy:.3f}")
            self.is_trained = True

            # Save final models metadata
            self.save_final_models_metadata(progress['accuracies'])

        return progress['accuracies']

    def save_final_models_metadata(self, accuracies):
        """Save final metadata when all models are trained"""
        metadata = {
            'n': self.n,
            'q': self.q,
            'num_models': self.n,
            'valid_models': len([acc for acc in accuracies if acc > 0]),
            'accuracies': accuracies,
            'tensorflow_version': tf.__version__,
            'model_indices': [i for i in range(self.n)]
        }

        with open(self.model_dir / "metadata.json", 'w') as f:
            json.dump(metadata, f, indent=2)

        print(f"Final metadata saved!")

    def load_models(self):
        """Load all trained models from disk"""
        metadata_path = self.model_dir / "metadata.json"

        if not metadata_path.exists():
            # Try loading from training progress
            progress = self.get_training_progress()
            if not progress['trained_models']:
                raise FileNotFoundError(f"No saved models found in {self.model_dir}/")

            trained_models = progress['trained_models']
            accuracies = progress['accuracies']
        else:
            # Load from final metadata
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)

            if metadata['n'] != self.n or metadata['q'] != self.q:
                raise ValueError(f"Saved models were trained with different parameters")

            trained_models = metadata['model_indices']
            accuracies = metadata['accuracies']

        print(f"Loading {len(trained_models)} trained models from {self.model_dir}/")

        # Load each model
        loaded_count = 0
        for idx in trained_models:
            model_path = self.model_dir / f"bit_model_{idx:03d}.h5"
            if model_path.exists():
                try:
                    self.models[idx] = keras.models.load_model(str(model_path))
                    loaded_count += 1
                except Exception as e:
                    print(f"Warning: Could not load model {idx}: {e}")
                    self.models[idx] = None

        self.is_trained = True
        print(f"Successfully loaded {loaded_count} models!")

        if accuracies:
            valid_accuracies = [acc for acc in accuracies if acc > 0]
            if valid_accuracies:
                avg_accuracy = np.mean(valid_accuracies)
                print(f"Average training accuracy: {avg_accuracy:.3f}")

        return accuracies

    def models_exist(self):
        """Check if saved models exist"""
        return (self.model_dir / "metadata.json").exists() or (self.model_dir / "training_progress.json").exists()

    def get_training_status(self):
        """Get detailed training status"""
        if not (self.model_dir / "training_progress.json").exists():
            return "No training started"

        progress = self.get_training_progress()
        trained_count = len(progress['trained_models'])
        total_models = self.n

        return f"Training progress: {trained_count}/{total_models} models ({trained_count/total_models:.1%})"

    def predict_secret(self, X_test):
        """Predict the secret key using trained models"""
        if not any(model is not None for model in self.models):
            raise ValueError("No models loaded! Call load_models() first.")

        predicted_secret = []
        confidences = []

        for i, model in enumerate(self.models):
            if model is not None:
                # Use the first test sample
                prob = model.predict(X_test[:1], verbose=0)[0][0]
                bit_prediction = 1 if prob > 0.5 else 0
                confidence = max(prob, 1 - prob)
            else:
                # If model not trained, make a random guess
                bit_prediction = np.random.randint(0, 2)
                confidence = 0.5

            predicted_secret.append(bit_prediction)
            confidences.append(confidence)

        return np.array(predicted_secret), np.array(confidences)

    def attack_ciphertext(self, a, b, predicted_secret):
        """Decrypt a ciphertext using the recovered secret"""
        # Compute dot product mod q
        dot_product = sum(a[i] * predicted_secret[i] for i in range(len(a))) % self.q

        # Recover phase
        phase = (b - dot_product + self.q) % self.q

        # Decode message
        message = round(phase / delta) % p
        return message

def encrypt_message(message, secret, noise_std=3.2):
    """Encrypt a message using LWE (for testing)"""
    a = np.random.randint(0, q, n)
    noise = int(np.round(np.random.normal(0, noise_std)))
    dot_product = np.dot(a, secret) % q
    b = (dot_product + delta * message + noise) % q
    return a.tolist(), b

def demonstrate_chunked_training():
    """Demonstrate chunked training workflow"""
    print("=== LWE Attack with Chunked Training ===\n")

    # Initialize attacker
    attacker = LWEAttacker(model_dir="lwe_chunked_models")

    # Check training status
    print(f"Training status: {attacker.get_training_status()}")

    # Check if we have training data
    try:
        X, secret = attacker.load_training_data()
        print(f"Found existing training data: {X.shape[0]} samples")
    except FileNotFoundError:
        print("No existing training data found. Generating new data...")

        # Generate data
        generator = LWEDataGenerator()
        try:
            true_secret, samples = generator.generate_data(num_samples=5000)
            print(f"Generated {len(samples)} samples")
        except Exception as e:
            print(f"Error generating data: {e}")
            print("Using simulated data instead...")
            true_secret, samples = simulate_lwe_data()

        # Prepare training data
        X, secret = create_training_data(true_secret, samples)
        print(f"Training data shape: {X.shape}")

    # Menu for user
    while True:
        print(f"\nCurrent status: {attacker.get_training_status()}")
        print("\nOptions:")
        print("1. Start/Continue chunked training")
        print("2. Load existing models and test")
        print("3. Show training progress")
        print("4. Exit")

        choice = input("Enter choice [1-4]: ").strip()

        if choice == "1":
            # Configure chunk size
            chunk_size = input("Enter chunk size (default 100): ").strip()
            chunk_size = int(chunk_size) if chunk_size else 100

            epochs = input("Enter epochs per model (default 30): ").strip()
            epochs = int(epochs) if epochs else 30

            print(f"\nStarting chunked training with chunk size {chunk_size}")
            print("You can stop at any time and resume later!")

            # Train models
            accuracies = attacker.train_models_chunked(X, secret, chunk_size=chunk_size, epochs=epochs)

        elif choice == "2":
            # Load models and test
            try:
                attacker.load_models()

                # Test the attack
                print("\nTesting attack on sample messages...")
                test_messages = [0, 1, 2, 3]

                for msg in test_messages:
                    a_test, b_test = encrypt_message(msg, secret)
                    predicted_secret, confidences = attacker.predict_secret(X)
                    decrypted_msg = attacker.attack_ciphertext(a_test, b_test, predicted_secret)

                    success = "✓" if decrypted_msg == msg else "✗"
                    print(f"Message {msg} -> Decrypted {decrypted_msg} {success}")

                # Show statistics
                correct_bits = sum(predicted_secret[i] == secret[i] for i in range(len(secret)))
                print(f"\nSecret recovery rate: {correct_bits}/{len(secret)} ({correct_bits/len(secret):.1%})")

            except Exception as e:
                print(f"Error loading models: {e}")

        elif choice == "3":
            # Show detailed progress
            progress = attacker.get_training_progress()
            trained_models = progress['trained_models']

            print(f"\nDetailed Progress:")
            print(f"Total models: {attacker.n}")
            print(f"Trained models: {len(trained_models)}")
            print(f"Completion: {len(trained_models)/attacker.n:.1%}")

            if trained_models:
                accuracies = [progress['accuracies'][i] for i in trained_models]
                print(f"Average accuracy: {np.mean(accuracies):.3f}")
                print(f"Models with >90% accuracy: {sum(acc > 0.9 for acc in accuracies)}")

        elif choice == "4":
            break

        else:
            print("Invalid choice!")

def simulate_lwe_data():
    """Simulate LWE data when C++ compilation fails"""
    print("Simulating LWE data...")

    # Generate random secret
    secret = np.random.randint(0, 2, n)

    # Generate samples
    samples = []
    for _ in range(5000):
        a = np.random.randint(0, q, n)
        noise = np.random.normal(0, 3.2)
        dot_product = np.dot(a, secret) % q
        b = (dot_product + int(noise)) % q
        samples.append((a.tolist(), b))

    return secret, samples

if __name__ == "__main__":
    demonstrate_chunked_training()
