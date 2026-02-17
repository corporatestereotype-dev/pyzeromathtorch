import numpy as np
from scipy.optimize import minimize_scalar  # For learning rate line search (optional purity boost)
import sympy as sp  # For symbolic FFZ torsion (finite-order computation)
import matplotlib.pyplot as plt  # For plateau visualization
from collections import Counter  # For bag-of-words
import re  # For basic tokenization
import os  # For path handling

# FFZ Helper: Symbolic Torsion Deformation (extrapolated from torsion algebra A(a,b))
def ffz_torsion_deform(vector, a=3, b=5, beta=0.1):
    """
    FFZ-inspired deformation: Model vector as torsion element in Z/aZ ⊕ Z/bZ.
    Use sympy symbolically for order and balance, then return to numpy for mean shift.
    """
    if len(vector) == 0:
        return vector.astype(float)

    # Symbolic Fibonacci orders (finite cancellation period)
    fib_n, fib_np1 = sp.fibonacci(a), sp.fibonacci(b)
    order = fib_n * fib_np1  # effective "torsion order"

    # Convert input vector to sympy Matrix
    sym_vec = sp.Matrix(vector.astype(float))  # ensure float for norm

    # Compute norm symbolically (or numerically — sympy norm is fine)
    norm = sym_vec.norm() + sp.Rational(1, 100000)  # avoid div-by-zero

    # Torsion balance term: -beta * unit vector * (norm / order)
    unit_vec = sym_vec / norm
    balance = -beta * unit_vec * (norm / order)

    # Apply deformation symbolically
    deformed_sym = sym_vec + balance

    # Convert BACK to numpy BEFORE doing mean subtraction
    deformed_np = np.array(deformed_sym).flatten().astype(float)

    # Enforce zero-sum invariant in numpy (fast & reliable)
    mean_val = np.mean(deformed_np)
    deformed_np -= mean_val

    # Optional: light clipping to prevent explosion (FFZ boundedness spirit)
    deformed_np = np.clip(deformed_np, -10.0, 10.0)

    return deformed_np

# Basic NLP Vectorizer (bag-of-words, no external libs)
def bag_of_words(texts, vocab_size=100):
    """
    Simple tokenizer and vectorizer: Regex split, count freq, truncate vocab.
    For NLP inference without frameworks.
    """
    all_words = []
    for text in texts:
        words = re.findall(r'\w+', text.lower())
        all_words.extend(words)
    vocab = [word for word, _ in Counter(all_words).most_common(vocab_size)]
    vectors = []
    for text in texts:
        words = re.findall(r'\w+', text.lower())
        vec = np.array([words.count(word) for word in vocab], dtype=float)
        vectors.append(vec / (np.linalg.norm(vec) + 1e-5))  # Normalize
    return np.array(vectors), vocab

# Vectorize using existing training vocabulary (critical for inference alignment)
def vectorize_with_vocab(texts, vocab):
    """
    Vectorize new texts using the exact same vocabulary order as training.
    Words not in vocab get count 0.
    """
    vectors = []
    for text in texts:
        words = re.findall(r'\w+', text.lower())
        vec = np.zeros(len(vocab), dtype=float)
        word_counts = Counter(words)
        for i, word in enumerate(vocab):
            vec[i] = word_counts[word]
        vec /= (np.linalg.norm(vec) + 1e-5)  # normalize
        vectors.append(vec)
    return np.array(vectors)

# Simple Perceptron Model (from scratch, numpy-only forward/backprop)
class FFZPerceptron:
    def __init__(self, input_size, learning_rate=0.01, beta=0.1):
        self.weights = np.random.randn(input_size) * 0.01  # Small init
        self.bias = 0.0
        self.lr = learning_rate
        self.beta = beta  # FFZ deformation strength

    def forward(self, x):
        z = np.dot(x, self.weights) + self.bias
        return 1 / (1 + np.exp(-z))  # Sigmoid activation

    def loss(self, y_pred, y_true):
        return -np.mean(y_true * np.log(y_pred + 1e-5) + (1 - y_true) * np.log(1 - y_pred + 1e-5))  # BCE

    def train_step(self, x, y):
        y_pred = self.forward(x)
        error = y_pred - y
        grad_w = np.dot(x.T, error) / len(y)  # Gradient
        # After grad_w = ...
        grad_w += 0.0005 * self.weights  # tiny L2 on weights]
        grad_b = np.mean(error)

        # Apply FFZ torsion deformation to gradients (stabilization)
        grad_w = ffz_torsion_deform(grad_w, beta=self.beta)

        if not np.all(np.isfinite(grad_w)):
            print("Warning: Non-finite gradients after FFZ deformation — clipping")
            grad_w = np.nan_to_num(grad_w, nan=0.0, posinf=1.0, neginf=-1.0)

        # Update with optional scipy line search for lr (purity enhancement)
        def lr_obj(alpha):
            temp_w = self.weights - alpha * grad_w
            temp_b = self.bias - alpha * grad_b
            temp_pred = 1 / (1 + np.exp(- (np.dot(x, temp_w) + temp_b)))
            return self.loss(temp_pred, y)
        optimal_lr = minimize_scalar(lr_obj, bounds=(0, self.lr * 2), method='bounded').x

        self.weights -= optimal_lr * grad_w
        self.bias -= optimal_lr * grad_b
        return self.loss(y_pred, y)

# Generate Synthetic NLP Data (positive/negative phrases for binary classification)
def generate_synthetic_data(n_samples=100, input_size=50):
    # Positive/negative word pools (simple NLP proxy)
    sentence_words = np.random.choice(pos_words if is_pos else neg_words, size=25)
    # Optional: repeat 2–3 key sentiment words to amplify signal
    if np.random.rand() > 0.6:
        key_word = np.random.choice(pos_words if is_pos else neg_words)
        sentence_words = np.append(sentence_words, [key_word] * np.random.randint(2,5))
    sentence = ' '.join(sentence_words)
    pos_words = ['great', 'amazing', 'love', 'excellent', 'fantastic', 'wonderful', 'superb', 'brilliant', 'outstanding', 'terrific']
    neg_words = ['bad', 'terrible', 'hate', 'poor', 'awful', 'horrible', 'dreadful', 'pathetic', 'useless', 'disappointing']
    test_texts = []
    labels = []
    for _ in range(n_samples):
        is_pos = np.random.rand() > 0.5
        sentence = ' '.join(np.random.choice(pos_words if is_pos else neg_words, size=14))
        texts.append(sentence)
        labels.append(1 if is_pos else 0)
    X, vocab = bag_of_words(texts, vocab_size=input_size)
    return X, np.array(labels), vocab

# ──────────────────────────────────────────────────────────────────────────────
# Main Training and Inference
# ──────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    # ── Data Generation ───────────────────────────────────────────────────────
    VOCAB_SIZE = 30
    X_train, y_train, vocab = generate_synthetic_data(200, input_size=VOCAB_SIZE)
   
    print(f"X_train shape: {X_train.shape}, actual vocab size: {len(vocab)}")
    assert X_train.shape[1] == len(vocab), "Vectorizer inconsistency"
    print(f"X_train shape: {X_train.shape}, expected features: {len(vocab)}")

    # Initialize model with ACTUAL feature count
    model = FFZPerceptron(input_size=X_train.shape[1], beta=0.05)

    # ── Training Loop ─────────────────────────────────────────────────────────
    epochs = 100
    losses = []
    for epoch in range(epochs):
        loss = model.train_step(X_train, y_train)
        losses.append(loss)
        if epoch % 10 == 0:
            print(f"Epoch {epoch}, Loss: {loss:.4f}")

    # ── Save Plot (Codespace-friendly — no plt.show()) ────────────────────────
    plt.title("FFZ Torsion-Stabilized Loss Curve\nPure NumPy Perceptron – Binary Sentiment", fontsize=14, fontweight='bold')
    plt.figtext(0.99, 0.01, "β=0.05 • 100 epochs • vocab=20", ha="right", va="bottom", fontsize=8, alpha=0.6)
    plt.figure(figsize=(10, 6))
    plt.plot(losses, label='Training Loss', color='#1f77b4', linewidth=2.5)
    plt.title("Training Loss Curve (FFZ Stabilized)", fontsize=14, fontweight='bold')
    plt.xlabel("Epoch", fontsize=12)
    plt.ylabel("BCE Loss", fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()
    plt.tight_layout()

    # Save files
    plot_png = "ffz_loss_curve.png"
    plot_pdf = "ffz_loss_curve.pdf"
    plt.savefig(plot_png, dpi=300, bbox_inches='tight')
    plt.savefig(plot_pdf, bbox_inches='tight')
    plt.close()

    print(f"\nLoss curve saved:")
    print(f"  • {plot_png}   (PNG — best for quick view)")
    print(f"  • {plot_pdf}  (PDF — best for papers/GitHub)")
    print("\n→ In Codespace file explorer (left sidebar): right-click file → Download")
    print("   Or click file to preview in browser → right-click image → Save as...")

    # ── Save Additional Results for Download ──────────────────────────────────
    np.save("training_losses.npy", np.array(losses))

    with open("final_results_summary.txt", "w") as f:
        f.write(f"Final Loss (epoch {epochs}): {losses[-1]:.4f}\n\n")
        f.write("Test Texts & Predictions:\n")
        # We'll fill this after inference

    # ── Inference ─────────────────────────────────────────────────────────────
    test_texts = [
        "This is amazing and great and wonderful and fantastic!",
        "This is terrible and bad and horrible and awful!",
        "I really love this superb brilliant thing so much.",
        "What a pathetic useless disappointing dreadful mess."
        "This is amazing and great!",
        "This is terrible and bad.",
        "I love this wonderful thing!",
        "What a horrible disappointing mess."
    ]

    X_test = vectorize_with_vocab(test_texts, vocab)
    print(f"X_test shape: {X_test.shape}")

    preds = model.forward(X_test)
    print("\nInference Predictions (Positive Probability):")
    for txt, prob in zip(test_texts, preds):
        print(f"  '{txt}' → {prob:.4f}")

    # Append predictions to summary file
    with open("final_results_summary.txt", "a") as f:
        for txt, prob in zip(test_texts, preds):
            f.write(f"  '{txt}' → {prob:.4f}\n")

    print("\nResults also saved to:")
    print("  • training_losses.npy")
    print("  • final_results_summary.txt")
    print("\nAll files ready for download in Codespace file explorer.")
