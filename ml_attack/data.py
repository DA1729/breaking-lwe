import numpy as np
import subprocess
import tempfile
import os

n = 512
q = 12289
p = 4
delta = q // p

cpp_code = '''
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
    
    std::cout << "SECRET:";
    for (int si : s) std::cout << " " << si;
    std::cout << std::endl;
    
    for (int i = 0; i < 10000; ++i) {
        auto ct = encrypt(0, s, gen);
        std::cout << "SAMPLE:";
        for (int ai : ct.first) std::cout << " " << ai;
        std::cout << " " << ct.second << std::endl;
    }
    return 0;
}
'''

def generate_and_save():
    with tempfile.NamedTemporaryFile(mode='w', suffix='.cpp', delete=False) as f:
        f.write(cpp_code)
        cpp_file = f.name

    exe_file = cpp_file.replace('.cpp', '')
    subprocess.run(['g++', '-o', exe_file, cpp_file], check=True)
    result = subprocess.run([exe_file], capture_output=True, text=True, check=True)

    os.unlink(cpp_file)
    os.unlink(exe_file)

    lines = result.stdout.strip().split('\n')
    secret_line = [line for line in lines if line.startswith('SECRET:')][0]
    secret = np.array(list(map(int, secret_line.split()[1:])), dtype=np.uint8)

    samples = []
    for line in lines:
        if line.startswith('SAMPLE:'):
            parts = list(map(int, line.split()[1:]))
            a = parts[:-1]
            b = parts[-1]
            samples.append((a, b))

    X = np.array([a for (a, b) in samples], dtype=np.uint16)
    B = np.array([b for (a, b) in samples], dtype=np.uint16)

    np.savez_compressed("lwe_dataset.npz", X=X, B=B, secret=secret)
    print("✅ Data saved to lwe_dataset.npz")

if __name__ == "__main__":
    generate_and_save()

