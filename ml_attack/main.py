import numpy as np
import tensorflow as tf
from tensorflow import keras
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import subprocess
import tempfile
import os

n = 512
q = 12289
p = 4
delta = q // p

class lwe_data_generator:
    def __init__(self, cpp_executable_path = None):
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

    def generate_data(self, num_samples = 10000):
        """generate lwe samples and secret key"""

        with tempfile.NamedTemporaryFile(mode='w', suffix='.cpp', delete=False) as f:
            f.write(self.cpp_code)
            cpp_file = f.name

        try:
            # compile
            exe_file = cpp_file.replace('.cpp', '')
            subprocess.run(['g++', '-o', exe_file, cpp_file], check=True)

            # run and capture output
            result = subprocess.run([exe_file], capture_output=True, text=True, check=True)

            # parse output
            lines = result.stdout.strip().split('\n')

            # extract secret key
            secret_line = [line for line in lines if line.startswith('SECRET:')][0]
            secret = list(map(int, secret_line.split()[1:]))

            # extract samples
            sample_lines = [line for line in lines if line.startswith('SAMPLE:')]
            samples = []
            for line in sample_lines:
                parts = list(map(int, line.split()[1:]))
                a = parts[:-1]  # first n elements
                b = parts[-1]   # last element
                samples.append((a, b))

            return np.array(secret), samples

        finally:
            # cleanup
            for file in [cpp_file, exe_file]:
                if os.path.exists(file):
                    os.unlink(file)

def create_training_data(secret, samples):
    """convert LWE samples to ML training data"""
    X = []  # input vectors (a values)
    y = []  # target labels (secret bits)

    for a, b in samples:
        X.append(a)

    return np.array(X), np.array(secret)


class lwe_attacker:
    """machine learning attack on lwe"""

    def __init__(self, n = 512, q = 12289):
        self.n = n
        self.q = q
        self.models = []

    def create_bit_model(self):
        """neural network to predict one secret bit"""
        model = keras.Sequential([
            keras.layers.Dense(256, activation = 'relu', input_shape = (self.n)),
            keras.layers.Dropout(0.3),
            keras.layers.Dense(128, activation = 'relu'),
            keras.layers.Dropout(0.3),
            keras.layers.Dense(64, activation = 'relu'),
            keras.layers.Dense(1, activation = 'sigmoid')
        ])

        model.compile(
            optimizer = 'adam',
            loss = 'binary_crossentropy',
            metric = ['accuracy']
        )

        return model
