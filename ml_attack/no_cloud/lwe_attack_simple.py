#!/usr/bin/env python3
"""
LWE Attack Demonstration with Different Key

This script demonstrates what happens when you try to attack an LWE key
that is DIFFERENT from the one the models were trained on.
"""

import numpy as np
import os
import sys
import json
import pickle
from pathlib import Path
import time
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter

# Force TensorFlow to use CPU only
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf
from tensorflow import keras

# Set up matplotlib for better plots
plt.style.use('seaborn-v0_8' if 'seaborn-v0_8' in plt.style.available else 'default')
sns.set_palette("husl")

# LWE Parameters
n = 512
q = 12289
p = 4
delta = q // p

def load_models_sample(model_dir, num_models=50):
    """Load a sample of trained models"""
    models = {}
    model_dir = Path(model_dir)

    print(f"Loading {num_models} sample models from {model_dir}")

    for i in range(num_models):
        model_path = model_dir / f"bit_model_{i:03d}.h5"
        if model_path.exists():
            try:
                models[i] = keras.models.load_model(str(model_path))
                print(f"Loaded model {i}", end='\r')
            except Exception as e:
                print(f"Failed to load model {i}: {e}")

    print(f"\nSuccessfully loaded {len(models)} models")
    return models

def load_original_data(model_dir):
    """Load original training data"""
    data_path = Path(model_dir) / "training_data.pkl"
    with open(data_path, 'rb') as f:
        data = pickle.load(f)
    return data['X'], data['secret']

def generate_new_secret_key():
    """Generate a completely new secret key (different from training key)"""
    # Use a different random seed to ensure different key
    np.random.seed(42)  # Fixed seed for reproducibility
    new_secret = np.random.randint(0, 2, n)

    print(f"Generated NEW secret key (first 20 bits): {new_secret[:20]}")
    return new_secret

def encrypt_message(message, secret_key, noise_std=3.2):
    """Encrypt a message using LWE"""
    a = np.random.randint(0, q, n)
    noise = int(np.round(np.random.normal(0, noise_std)))
    dot_product = np.dot(a, secret_key) % q
    b = (dot_product + delta * message + noise) % q
    return a, b

def predict_secret_bits(models, sample_a):
    """Predict secret bits using loaded models"""
    predicted_bits = {}
    confidences = {}

    input_data = sample_a.reshape(1, -1)

    for i, model in models.items():
        try:
            prob = model.predict(input_data, verbose=0)[0][0]
            bit_prediction = 1 if prob > 0.5 else 0
            confidence = max(prob, 1 - prob)

            predicted_bits[i] = bit_prediction
            confidences[i] = confidence

        except Exception as e:
            print(f"Error predicting with model {i}: {e}")

    return predicted_bits, confidences

def create_key_comparison_plot(training_secret, actual_secret, predicted_bits, confidences, save_path=None):
    """Create comprehensive visualization of key comparison"""
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('LWE Attack Analysis: Training Key vs Actual Key vs Predictions', fontsize=16)

    # Plot 1: Bit-by-bit comparison for first 100 bits
    ax1 = axes[0, 0]
    positions = list(range(min(100, len(training_secret))))

    ax1.plot(positions, training_secret[:100], 'b-', label='Training Key', linewidth=2, alpha=0.7)
    ax1.plot(positions, actual_secret[:100], 'r-', label='Actual Key', linewidth=2, alpha=0.7)

    # Plot predictions where available
    pred_positions = [i for i in positions if i in predicted_bits]
    pred_values = [predicted_bits[i] for i in pred_positions]
    ax1.scatter(pred_positions, pred_values, c='green', s=30, label='Predictions', alpha=0.8, zorder=5)

    ax1.set_xlabel('Bit Position')
    ax1.set_ylabel('Bit Value')
    ax1.set_title('Bit-by-Bit Comparison (First 100 bits)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(-0.1, 1.1)

    # Plot 2: Accuracy comparison
    ax2 = axes[0, 1]
    if predicted_bits:
        training_matches = [1 if training_secret[i] == predicted_bits[i] else 0 for i in predicted_bits.keys()]
        actual_matches = [1 if actual_secret[i] == predicted_bits[i] else 0 for i in predicted_bits.keys()]

        training_acc = np.mean(training_matches)
        actual_acc = np.mean(actual_matches)
        random_acc = 0.5

        categories = ['Training Key', 'Actual Key', 'Random Guess']
        accuracies = [training_acc, actual_acc, random_acc]
        colors = ['blue', 'red', 'gray']

        bars = ax2.bar(categories, accuracies, color=colors, alpha=0.7)
        ax2.set_ylabel('Accuracy')
        ax2.set_title('Prediction Accuracy Comparison')
        ax2.set_ylim(0, 1)

        # Add value labels on bars
        for bar, acc in zip(bars, accuracies):
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                    f'{acc:.1%}', ha='center', va='bottom', fontweight='bold')

        # Add horizontal line at 50% (random)
        ax2.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5, label='Random (50%)')

    ax2.grid(True, alpha=0.3)

    # Plot 3: Confidence distribution
    ax3 = axes[1, 0]
    if confidences:
        conf_values = list(confidences.values())
        ax3.hist(conf_values, bins=20, alpha=0.7, color='purple', edgecolor='black')
        ax3.set_xlabel('Confidence Score')
        ax3.set_ylabel('Frequency')
        ax3.set_title('Prediction Confidence Distribution')
        ax3.axvline(x=np.mean(conf_values), color='red', linestyle='--',
                   label=f'Mean: {np.mean(conf_values):.3f}')
        ax3.legend()
        ax3.grid(True, alpha=0.3)

    # Plot 4: Key difference heatmap
    ax4 = axes[1, 1]
    n_display = min(200, len(training_secret))
    key_diff_matrix = np.zeros((2, n_display))

    key_diff_matrix[0, :] = training_secret[:n_display]
    key_diff_matrix[1, :] = actual_secret[:n_display]

    im = ax4.imshow(key_diff_matrix, cmap='RdYlBu', aspect='auto')
    ax4.set_xlabel('Bit Position')
    ax4.set_ylabel('Key Type')
    ax4.set_title(f'Key Comparison Heatmap (First {n_display} bits)')
    ax4.set_yticks([0, 1])
    ax4.set_yticklabels(['Training', 'Actual'])

    # Add colorbar
    cbar = plt.colorbar(im, ax=ax4)
    cbar.set_label('Bit Value')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Key comparison plot saved to {save_path}")

    plt.show()

def create_attack_results_plot(test_results, save_path=None):
    """Create visualization of attack results across different messages"""
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('LWE Attack Results Analysis', fontsize=16)

    messages = list(test_results.keys())
    training_accs = [test_results[msg]['training_accuracy'] for msg in messages]
    actual_accs = [test_results[msg]['actual_accuracy'] for msg in messages]
    confidences = [test_results[msg]['avg_confidence'] for msg in messages]
    decryption_success = [test_results[msg]['decryption_success'] for msg in messages]

    # Plot 1: Accuracy by message
    ax1 = axes[0, 0]
    x = np.arange(len(messages))
    width = 0.35

    bars1 = ax1.bar(x - width/2, training_accs, width, label='Training Key', alpha=0.7, color='blue')
    bars2 = ax1.bar(x + width/2, actual_accs, width, label='Actual Key', alpha=0.7, color='red')

    ax1.set_xlabel('Test Message')
    ax1.set_ylabel('Prediction Accuracy')
    ax1.set_title('Prediction Accuracy by Message')
    ax1.set_xticks(x)
    ax1.set_xticklabels([f'Msg {msg}' for msg in messages])
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5, label='Random')

    # Add value labels
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                    f'{height:.1%}', ha='center', va='bottom', fontsize=9)

    # Plot 2: Confidence by message
    ax2 = axes[0, 1]
    bars = ax2.bar(messages, confidences, alpha=0.7, color='purple')
    ax2.set_xlabel('Test Message')
    ax2.set_ylabel('Average Confidence')
    ax2.set_title('Prediction Confidence by Message')
    ax2.grid(True, alpha=0.3)

    for bar, conf in zip(bars, confidences):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{conf:.3f}', ha='center', va='bottom', fontsize=9)

    # Plot 3: Decryption success rate
    ax3 = axes[1, 0]
    success_count = sum(decryption_success)
    failure_count = len(decryption_success) - success_count

    colors = ['green' if success else 'red' for success in decryption_success]
    bars = ax3.bar(messages, [1 if success else 0 for success in decryption_success],
                   color=colors, alpha=0.7)
    ax3.set_xlabel('Test Message')
    ax3.set_ylabel('Decryption Success')
    ax3.set_title('Decryption Success by Message')
    ax3.set_ylim(0, 1.1)
    ax3.grid(True, alpha=0.3)

    for bar, success in zip(bars, decryption_success):
        ax3.text(bar.get_x() + bar.get_width()/2., 0.5,
                '✓' if success else '✗', ha='center', va='center',
                fontsize=20, fontweight='bold')

    # Plot 4: Summary statistics
    ax4 = axes[1, 1]
    ax4.axis('off')

    # Create summary text
    summary_text = f"""
    ATTACK SUMMARY

    Total Messages Tested: {len(messages)}

    Prediction Accuracy:
    • Training Key: {np.mean(training_accs):.1%} ± {np.std(training_accs):.1%}
    • Actual Key: {np.mean(actual_accs):.1%} ± {np.std(actual_accs):.1%}

    Average Confidence: {np.mean(confidences):.3f}

    Decryption Success: {success_count}/{len(messages)} ({success_count/len(messages):.1%})

    CONCLUSION:
    {'✓ Attack Failed (LWE Secure)' if success_count == 0 else '✗ Attack Succeeded'}

    The models can only predict the training key
    but fail on the actual (different) key.
    """

    ax4.text(0.1, 0.9, summary_text, transform=ax4.transAxes, fontsize=12,
             verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray", alpha=0.8))

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Attack results plot saved to {save_path}")

    plt.show()

def create_key_difference_analysis(training_secret, actual_secret, save_path=None):
    """Create detailed analysis of key differences"""
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('Key Difference Analysis', fontsize=16)

    # Calculate differences
    differences = training_secret != actual_secret
    diff_positions = np.where(differences)[0]
    same_positions = np.where(~differences)[0]

    # Plot 1: Difference distribution
    ax1 = axes[0, 0]
    chunk_size = 32
    num_chunks = len(training_secret) // chunk_size
    chunk_diffs = []

    for i in range(num_chunks):
        start = i * chunk_size
        end = start + chunk_size
        chunk_diff = np.sum(differences[start:end])
        chunk_diffs.append(chunk_diff)

    ax1.bar(range(num_chunks), chunk_diffs, alpha=0.7, color='orange')
    ax1.set_xlabel('Chunk Index')
    ax1.set_ylabel('Number of Differences')
    ax1.set_title(f'Key Differences by Chunk (chunks of {chunk_size} bits)')
    ax1.grid(True, alpha=0.3)

    # Plot 2: Difference pattern
    ax2 = axes[0, 1]
    diff_binary = differences.astype(int)

    # Reshape for visualization
    rows = 16
    cols = len(training_secret) // rows
    diff_matrix = diff_binary[:rows*cols].reshape(rows, cols)

    im = ax2.imshow(diff_matrix, cmap='RdYlBu_r', aspect='auto')
    ax2.set_xlabel('Bit Position (grouped)')
    ax2.set_ylabel('Row')
    ax2.set_title('Key Difference Pattern')

    # Add colorbar
    cbar = plt.colorbar(im, ax=ax2)
    cbar.set_label('Different (1) / Same (0)')

    # Plot 3: Hamming distance distribution
    ax3 = axes[1, 0]
    total_differences = np.sum(differences)
    hamming_distance = total_differences / len(training_secret)

    ax3.bar(['Same Bits', 'Different Bits'],
            [len(same_positions), len(diff_positions)],
            color=['green', 'red'], alpha=0.7)
    ax3.set_ylabel('Count')
    ax3.set_title('Overall Key Difference Summary')
    ax3.grid(True, alpha=0.3)

    # Add value labels
    ax3.text(0, len(same_positions) + 10, f'{len(same_positions)}\n({len(same_positions)/len(training_secret):.1%})',
             ha='center', va='bottom', fontweight='bold')
    ax3.text(1, len(diff_positions) + 10, f'{len(diff_positions)}\n({len(diff_positions)/len(training_secret):.1%})',
             ha='center', va='bottom', fontweight='bold')

    # Plot 4: Statistical summary
    ax4 = axes[1, 1]
    ax4.axis('off')

    # Calculate some statistics
    expected_diff = 0.5  # Expected for random keys
    actual_diff = hamming_distance

    stats_text = f"""
    KEY DIFFERENCE STATISTICS

    Total Key Length: {len(training_secret)} bits

    Same Bits: {len(same_positions)} ({len(same_positions)/len(training_secret):.1%})
    Different Bits: {len(diff_positions)} ({len(diff_positions)/len(training_secret):.1%})

    Hamming Distance: {total_differences}
    Normalized Distance: {hamming_distance:.3f}

    Expected for Random Keys: ~0.500
    Actual Difference: {actual_diff:.3f}

    INTERPRETATION:
    {'Keys are significantly different' if actual_diff > 0.3 else 'Keys are similar'}

    This level of difference should make
    the ML attack ineffective on the new key.
    """

    ax4.text(0.1, 0.9, stats_text, transform=ax4.transAxes, fontsize=11,
             verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue", alpha=0.8))

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Key difference analysis saved to {save_path}")

    plt.show()

    return {
        'total_differences': total_differences,
        'hamming_distance': hamming_distance,
        'same_positions': len(same_positions),
        'diff_positions': len(diff_positions)
    }

def decrypt_message(a, b, predicted_secret):
    """Decrypt using predicted secret"""
    dot_product = np.dot(a, predicted_secret) % q
    noisy_message = (b - dot_product) % q
    message = round(noisy_message / delta) % p
    return message

def display_secret_comparison(original_secret, predicted_bits, confidences, actual_secret, num_display=30):
    """Display detailed comparison between training secret, actual secret, and predicted bits"""
    print(f"\n=== Secret Key Comparison (First {num_display} bits) ===")
    print("Bit  | Training | Actual | Predicted | Confidence | Match Training | Match Actual")
    print("-" * 80)

    correct_training = 0
    correct_actual = 0
    total_count = 0

    for i in range(min(num_display, len(original_secret))):
        training_bit = original_secret[i]
        actual_bit = actual_secret[i]

        if i in predicted_bits:
            predicted_bit = predicted_bits[i]
            confidence = confidences[i]
            match_training = "✓" if training_bit == predicted_bit else "✗"
            match_actual = "✓" if actual_bit == predicted_bit else "✗"

            if training_bit == predicted_bit:
                correct_training += 1
            if actual_bit == predicted_bit:
                correct_actual += 1
            total_count += 1

            print(f"{i:3d}  |    {training_bit}     |   {actual_bit}    |     {predicted_bit}     |   {confidence:.3f}    |       {match_training}        |      {match_actual}")
        else:
            print(f"{i:3d}  |    {training_bit}     |   {actual_bit}    |     ?     |     ?      |       ?        |      ?")

    if total_count > 0:
        training_accuracy = correct_training / total_count
        actual_accuracy = correct_actual / total_count
        print(f"\nAccuracy vs Training Key: {correct_training}/{total_count} ({training_accuracy:.1%})")
        print(f"Accuracy vs Actual Key:   {correct_actual}/{total_count} ({actual_accuracy:.1%})")

    return correct_training, correct_actual, total_count

def main():
    """Main demonstration"""
    print("=== LWE Machine Learning Attack Demo - DIFFERENT KEY ===\n")

    model_dir = "lwe_chunked_models"

    # Load sample models
    models = load_models_sample(model_dir, num_models=50)
    if not models:
        print("No models loaded!")
        return 1

    # Load original training data
    X, training_secret = load_original_data(model_dir)
    print(f"Training secret (first 20 bits): {training_secret[:20]}")

    # Generate a DIFFERENT secret key for testing
    actual_secret = generate_new_secret_key()

    # Create key difference analysis
    print("\n=== Analyzing Key Differences ===")
    key_diff_stats = create_key_difference_analysis(training_secret, actual_secret,
                                                   save_path="key_difference_analysis.png")

    # Show key differences
    print(f"\n=== Key Comparison ===")
    print("Position:  ", end="")
    for i in range(20):
        print(f"{i:2d} ", end="")
    print()

    print("Training:  ", end="")
    for i in range(20):
        print(f"{training_secret[i]:2d} ", end="")
    print()

    print("Actual:    ", end="")
    for i in range(20):
        print(f"{actual_secret[i]:2d} ", end="")
    print()

    print("Different: ", end="")
    for i in range(20):
        different = "✗" if training_secret[i] != actual_secret[i] else " "
        print(f"{different:2s} ", end="")
    print()

    # Calculate how different the keys are
    differences = sum(1 for i in range(n) if training_secret[i] != actual_secret[i])
    print(f"\nKeys differ in {differences}/{n} positions ({differences/n:.1%})")

    # Test messages
    test_messages = [0, 1, 2, 3]
    test_results = {}

    print(f"\n=== Testing Attack on Messages {test_messages} (with DIFFERENT key) ===")

    for message in test_messages:
        print(f"\n--- Message {message} ---")

        # Encrypt message with the NEW key (not training key)
        a, b = encrypt_message(message, actual_secret)
        print(f"Encrypted ciphertext: b = {b}")

        # Predict secret bits using models trained on OLD key
        start_time = time.time()
        predicted_bits, confidences = predict_secret_bits(models, a)
        prediction_time = time.time() - start_time

        print(f"Predicted {len(predicted_bits)} secret bits in {prediction_time:.3f}s")

        # Check accuracy of predictions against BOTH keys
        correct_training = sum(1 for i, bit in predicted_bits.items() if training_secret[i] == bit)
        correct_actual = sum(1 for i, bit in predicted_bits.items() if actual_secret[i] == bit)

        training_accuracy = correct_training / len(predicted_bits)
        actual_accuracy = correct_actual / len(predicted_bits)
        avg_confidence = np.mean(list(confidences.values()))

        print(f"Bit prediction accuracy vs training key: {correct_training}/{len(predicted_bits)} ({training_accuracy:.1%})")
        print(f"Bit prediction accuracy vs actual key:   {correct_actual}/{len(predicted_bits)} ({actual_accuracy:.1%})")
        print(f"Average confidence: {avg_confidence:.3f}")

        # Display detailed comparison
        display_secret_comparison(training_secret, predicted_bits, confidences, actual_secret, num_display=20)

        # Try decryption with predicted secret
        full_predicted_secret = np.random.randint(0, 2, n)
        for i, bit in predicted_bits.items():
            full_predicted_secret[i] = bit

        decrypted = decrypt_message(a, b, full_predicted_secret)
        success = decrypted == message

        print(f"Decryption result: {decrypted} {'✓' if success else '✗'}")

        # Also try with the actual secret (should work)
        correct_decrypted = decrypt_message(a, b, actual_secret)
        print(f"Correct decryption (with actual key): {correct_decrypted} {'✓' if correct_decrypted == message else '✗'}")

        # Store results for plotting
        test_results[message] = {
            'training_accuracy': training_accuracy,
            'actual_accuracy': actual_accuracy,
            'avg_confidence': avg_confidence,
            'decryption_success': success,
            'predicted_bits': predicted_bits,
            'confidences': confidences
        }

    # Create comprehensive visualizations
    print("\n=== Creating Visualizations ===")

    # Use the last message's predictions for key comparison plot
    last_predictions = test_results[test_messages[-1]]['predicted_bits']
    last_confidences = test_results[test_messages[-1]]['confidences']

    create_key_comparison_plot(training_secret, actual_secret, last_predictions,
                              last_confidences, save_path="key_comparison_analysis.png")

    create_attack_results_plot(test_results, save_path="attack_results_analysis.png")

    print(f"\n=== Final Analysis ===")

    # Show overall secret key comparison
    print("\n--- Complete Secret Key Overview ---")
    print(f"Training secret key (first 50 bits):")
    print(' '.join(f"{training_secret[i]}" for i in range(50)))

    print(f"Actual secret key (first 50 bits):")
    print(' '.join(f"{actual_secret[i]}" for i in range(50)))

    # Create full predicted secret with available predictions
    full_predicted = ['?' for _ in range(50)]
    for i, bit in last_predictions.items():
        if i < 50:
            full_predicted[i] = str(bit)

    print(f"Predicted secret key (first 50 bits):")
    print(' '.join(full_predicted))

    # Show which positions match training vs actual
    training_match = []
    actual_match = []
    for i in range(50):
        if i in last_predictions:
            training_match.append('✓' if training_secret[i] == last_predictions[i] else '✗')
            actual_match.append('✓' if actual_secret[i] == last_predictions[i] else '✗')
        else:
            training_match.append('?')
            actual_match.append('?')

    print(f"Match vs training key (first 50 bits):")
    print(' '.join(training_match))

    print(f"Match vs actual key (first 50 bits):")
    print(' '.join(actual_match))

    print(f"\n=== Attack Results Summary ===")
    print(f"Models loaded: {len(models)}/512 ({len(models)/512:.1%})")
    print(f"Key difference: {differences}/{n} bits ({differences/n:.1%})")

    # Calculate final accuracies across all messages
    all_training_accs = [test_results[msg]['training_accuracy'] for msg in test_messages]
    all_actual_accs = [test_results[msg]['actual_accuracy'] for msg in test_messages]
    all_confidences = [test_results[msg]['avg_confidence'] for msg in test_messages]
    successful_decryptions = sum(test_results[msg]['decryption_success'] for msg in test_messages)

    print(f"Average prediction accuracy vs training key: {np.mean(all_training_accs):.1%}")
    print(f"Average prediction accuracy vs actual key: {np.mean(all_actual_accs):.1%}")
    print(f"Average confidence: {np.mean(all_confidences):.3f}")
    print(f"Successful decryptions: {successful_decryptions}/{len(test_messages)} ({successful_decryptions/len(test_messages):.1%})")

    print(f"\n=== Key Insights ===")
    print(f"• Models trained on one key perform poorly on different keys")
    print(f"• Prediction accuracy on new key ≈ {np.mean(all_actual_accs):.1%} (close to random 50%)")
    print(f"• Models still remember training key patterns: {np.mean(all_training_accs):.1%} accuracy")
    print(f"• This demonstrates that the attack doesn't generalize across keys")
    print(f"• LWE remains secure against this type of ML attack")

    print(f"\n=== Graphical Analysis Complete ===")
    print(f"Generated plots:")
    print(f"• key_difference_analysis.png - Key difference patterns")
    print(f"• key_comparison_analysis.png - Detailed key comparison")
    print(f"• attack_results_analysis.png - Attack performance summary")

    return 0

if __name__ == "__main__":
    sys.exit(main())
