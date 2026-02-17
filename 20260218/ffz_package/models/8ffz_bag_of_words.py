import numpy as np
from scipy.optimize import minimize_scalar
import sympy as sp
import matplotlib.pyplot as plt
from collections import Counter
import re
import os

# ── FFZ Torsion Deformation ──────────────────────────────────────────────────
def ffz_torsion_deform(vector, a=3, b=5, beta=0.1):
    if len(vector) == 0:
        return vector.astype(float)
    fib_n, fib_np1 = sp.fibonacci(a), sp.fibonacci(b)
    order = fib_n * fib_np1
    sym_vec = sp.Matrix(vector.astype(float))
    norm = sym_vec.norm() + sp.Rational(1, 100000)
    unit_vec = sym_vec / norm
    balance = -beta * unit_vec * (norm / order)
    deformed_sym = sym_vec + balance
    deformed_np = np.array(deformed_sym).flatten().astype(float)
    deformed_np -= np.mean(deformed_np)
    deformed_np = np.clip(deformed_np, -10.0, 10.0)
    return deformed_np

# ── Bag-of-Words Vectorizers ─────────────────────────────────────────────────
def bag_of_words(texts, vocab_size=100):
    all_words = []
    for text in texts:
        words = re.findall(r'\w+', text.lower())
        all_words.extend(words)
    vocab = [word for word, _ in Counter(all_words).most_common(vocab_size)]
    vectors = []
    for text in texts:
        words = re.findall(r'\w+', text.lower())
        vec = np.array([words.count(word) for word in vocab], dtype=float)
        vectors.append(vec / (np.linalg.norm(vec) + 1e-5))
    return np.array(vectors), vocab

def vectorize_with_vocab(texts, vocab):
    vectors = []
    for text in texts:
        words = re.findall(r'\w+', text.lower())
        vec = np.zeros(len(vocab), dtype=float)
        word_counts = Counter(words)
        for i, word in enumerate(vocab):
            vec[i] = word_counts[word]
        vec /= (np.linalg.norm(vec) + 1e-5)
        vectors.append(vec)
    return np.array(vectors)

# ── FFZ-Stabilized Neural Network (Single Hidden Layer) ─────────────────────
class FFZPerceptron:
    def __init__(self, input_size, hidden_size=16, learning_rate=0.5, beta=0.01):
        self.W1 = np.random.randn(input_size, hidden_size) * 0.01
        self.b1 = np.zeros(hidden_size)
        self.W2 = np.random.randn(hidden_size, 1) * 0.01
        self.b2 = 0.0
        self.lr = learning_rate
        self.beta = beta

    def forward(self, x):
        self.z1 = np.dot(x, self.W1) + self.b1
        self.a1 = 1 / (1 + np.exp(-self.z1))  # sigmoid hidden
        z2 = np.dot(self.a1, self.W2) + self.b2
        return 1 / (1 + np.exp(-z2))  # sigmoid output

    def loss(self, y_pred, y_true):
        return -np.mean(y_true * np.log(y_pred + 1e-5) + (1 - y_true) * np.log(1 - y_pred + 1e-5))

    def train_step(self, x, y):
        y_pred = self.forward(x)
        
        # CRITICAL FIX: Ensure y is column vector to match y_pred shape (batch_size, 1)
        if y.ndim == 1:
            y = y.reshape(-1, 1)

        # Backprop
        error_out = y_pred - y
        grad_W2 = np.dot(self.a1.T, error_out) / len(y)
        grad_b2 = np.mean(error_out)

        error_hidden = np.dot(error_out, self.W2.T) * self.a1 * (1 - self.a1)
        grad_W1 = np.dot(x.T, error_hidden) / len(y)
        grad_b1 = np.mean(error_hidden, axis=0)

        # Apply FFZ to all gradients (flatten, deform, reshape back)
        grad_W1 = ffz_torsion_deform(grad_W1.flatten(), beta=self.beta).reshape(grad_W1.shape)
        grad_W2 = ffz_torsion_deform(grad_W2.flatten(), beta=self.beta).reshape(grad_W2.shape)

        if not np.all(np.isfinite(grad_W1)) or not np.all(np.isfinite(grad_W2)):
            print("Warning: Non-finite gradients — clipping")
            grad_W1 = np.nan_to_num(grad_W1)
            grad_W2 = np.nan_to_num(grad_W2)

        # Simple GD update
        self.W1 -= self.lr * grad_W1
        self.b1 -= self.lr * grad_b1
        self.W2 -= self.lr * grad_W2
        self.b2 -= self.lr * grad_b2

        return self.loss(y_pred, y)

# ── Synthetic Data Generator ────────────────────────────────────────────────
def generate_synthetic_data(n_samples=200, input_size=30):
    pos_words = ['great', 'amazing', 'love', 'excellent', 'fantastic', 'wonderful', 'superb', 'brilliant', 'outstanding', 'terrific']
    neg_words = ['bad', 'terrible', 'hate', 'poor', 'awful', 'horrible', 'dreadful', 'pathetic', 'useless', 'disappointing']
    texts = []
    labels = []
    for _ in range(n_samples):
        is_pos = np.random.rand() > 0.5
        base_words = pos_words if is_pos else neg_words
        sentence_words = np.random.choice(base_words, size=25)
        if np.random.rand() > 0.6:
            key_word = np.random.choice(base_words)
            repeats = np.random.randint(2, 6)
            sentence_words = np.append(sentence_words, [key_word] * repeats)
        sentence = ' '.join(sentence_words)
        texts.append(sentence)
        labels.append(1 if is_pos else 0)
    X, vocab = bag_of_words(texts, vocab_size=input_size)
    return X, np.array(labels), vocab

# ── Main ─────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    VOCAB_SIZE = 30
    X_train, y_train, vocab = generate_synthetic_data(200, input_size=VOCAB_SIZE)
   
    print(f"X_train shape: {X_train.shape}, actual vocab size: {len(vocab)}")
    assert X_train.shape[1] == len(vocab), "Vectorizer inconsistency"
    print(f"X_train shape: {X_train.shape}, expected features: {len(vocab)}")

    model = FFZPerceptron(input_size=X_train.shape[1], hidden_size=16, beta=0.05)

    # epochs = 500
    epochs = 1000
    losses = []
    for epoch in range(epochs):
        loss = model.train_step(X_train, y_train)
        losses.append(loss)
        if epoch % 10 == 0:
            print(f"Epoch {epoch}, Loss: {loss:.4f}")

    # ── Plot & Save ───────────────────────────────────────────────────────────
    plt.figure(figsize=(10, 6))
    plt.plot(losses, label='Training Loss', color='#1f77b4', linewidth=2.5)
    plt.title("FFZ Torsion-Stabilized Loss Curve\nPure NumPy Perceptron – Binary Sentiment", fontsize=14, fontweight='bold')
    plt.xlabel("Epoch", fontsize=12)
    plt.ylabel("BCE Loss", fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()
    plt.figtext(0.99, 0.01, f"β={model.beta:.2f} • {epochs} epochs • vocab={len(vocab)}", 
                ha="right", va="bottom", fontsize=8, alpha=0.6)
    plt.tight_layout()

    plot_png = "ffz_loss_curve.png"
    plot_pdf = "ffz_loss_curve.pdf"
    plt.savefig(plot_png, dpi=300, bbox_inches='tight')
    plt.savefig(plot_pdf, bbox_inches='tight')
    plt.close()

    print(f"\nLoss curve saved: {plot_png} (PNG) • {plot_pdf} (PDF)")

    # ── Save Results ──────────────────────────────────────────────────────────
    np.save("training_losses.npy", np.array(losses))

    with open("final_results_summary.txt", "w") as f:
        f.write(f"Final Loss (epoch {epochs}): {losses[-1]:.4f}\n\n")
        f.write("Test Texts & Predictions:\n")

    # ── Hardcoded Test Examples ──────────────────────────────────────────────
    test_texts = [
        "This is amazing and great and wonderful and fantastic!",
        "This is terrible and bad and horrible and awful!",
        "I really love this superb brilliant thing so much.",
        "What a pathetic useless disappointing dreadful mess."
    ]

    X_test = vectorize_with_vocab(test_texts, vocab)
    print(f"\nX_test shape: {X_test.shape}")

    preds = model.forward(X_test).flatten()
    print("\nHardcoded Test Predictions (Positive Probability):")
    for txt, prob in zip(test_texts, preds):
        sentiment = "POSITIVE" if prob > 0.5 else "NEGATIVE"
        print(f"  '{txt}'")
        print(f"    → {prob:.4f} ({sentiment})\n")
        
        # Append to summary file
        with open("final_results_summary.txt", "a") as f:
            f.write(f"  '{txt}' → {prob:.4f} ({sentiment})\n")

    # ── INTERACTIVE USER PROMPT INTERFACE ────────────────────────────────────
    print("="*70)
    print("🤖 INTERACTIVE MODE: Enter text to analyze sentiment")
    print("   Commands:")
    print("   • Type 'quit', 'exit', or 'q' to stop")
    print("   • Type 'threshold <value>' to adjust decision boundary (default: 0.5)")
    print("   • Type 'vocab' to see the trained vocabulary")
    print("="*70)
    
    threshold = 0.5
    
    while True:
        try:
            user_input = input("\n> ").strip()
            
            if not user_input:
                continue
                
            cmd = user_input.lower()
            
            if cmd in ['quit', 'exit', 'q']:
                print("Exiting interactive mode...")
                break
                
            elif cmd == 'vocab':
                print(f"\nTrained vocabulary ({len(vocab)} words):")
                print(", ".join(vocab[:20]) + ("..." if len(vocab) > 20 else ""))
                continue
                
            elif cmd.startswith('threshold '):
                try:
                    new_thresh = float(cmd.split()[1])
                    threshold = max(0.0, min(1.0, new_thresh))
                    print(f"Decision threshold set to: {threshold}")
                except (IndexError, ValueError):
                    print("Usage: threshold <value> (e.g., threshold 0.3)")
                continue
            
            # Process text
            X_user = vectorize_with_vocab([user_input], vocab)
            pred = model.forward(X_user).flatten()[0]
            
            # Determine sentiment with confidence
            if pred >= threshold:
                sentiment = "POSITIVE 😊"
                confidence = pred
            else:
                sentiment = "NEGATIVE 😞"
                confidence = 1 - pred
            
            # Visual bar
            bar_length = 40
            fill_length = int(pred * bar_length)
            bar = "█" * fill_length + "░" * (bar_length - fill_length)
            
            print(f"\n  Prediction: {sentiment}")
            print(f"  Confidence: {confidence:.2%}")
            print(f"  Raw Score:  [{bar}] {pred:.4f}")
            
            # Warning if vocabulary might be insufficient
            words_in_text = set(re.findall(r'\w+', user_input.lower()))
            known_words = words_in_text & set(vocab)
            unknown_words = words_in_text - set(vocab)
            
            if unknown_words:
                print(f"  ⚠️  Unknown words (not in vocab): {','.join(list(unknown_words)[:5])}") 
                coverage = len(known_words) / len(words_in_text) * 100 if words_in_text else 0
                print(f"  Vocabulary coverage: {coverage:.1f}%")
            
        except KeyboardInterrupt:
            print("\nExiting...")
            break
        except Exception as e:
            print(f"Error processing input: {e}")

    print("\n" + "="*70)
    print("All files saved:")
    print("  • ffz_loss_curve.png / .pdf")
    print("  • training_losses.npy")
    print("  • final_results_summary.txt")
    print("="*70)
