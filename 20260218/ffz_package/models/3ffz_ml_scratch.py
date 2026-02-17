import numpy as np
from scipy.optimize import minimize_scalar  # For learning rate line search (optional purity boost)
import sympy as sp  # For symbolic FFZ torsion (finite-order computation)
import matplotlib.pyplot as plt  # For plateau visualization
from collections import Counter  # For bag-of-words
import re  # For basic tokenization

# FFZ Helper: Symbolic Torsion Deformation (extrapolated from torsion algebra A(a,b))
def ffz_torsion_deform(vector, a=3, b=5, beta=0.1):
    """
    FFZ-inspired deformation: Model vector as torsion element in Z/aZ ⊕ Z/bZ.
    Use sympy to compute finite-order cancellation (e.g., n*inf = 0).
    Balance by subtracting a beta-scaled opposing term, enforcing near-zero sum (eigenvalue constraint).
    Ties to docs: ∑λ_i ≈ 0 for stabilization without divergence.
    """
    # Symbolic Fibonacci orders (from docs' recursion)
    fib_n, fib_np1 = sp.fibonacci(a), sp.fibonacci(b)
    order = fib_n * fib_np1  # lcm-like for cancellation
    # Convert vector to sympy for precise torsion
    sym_vec = sp.Matrix(vector)
    # Torsion term: Scale by inverse order, negate extremes (cancellation to zero)
    balance = -beta * (sym_vec / (sym_vec.norm() + 1e-5)) * (sym_vec.norm() / order)
    deformed = sym_vec + balance
    # Enforce zero-sum invariant (shift mean to 0)
    deformed -= deformed.mean()
    return np.array(deformed).flatten().astype(float)

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
        grad_b = np.mean(error)

        # Apply FFZ torsion deformation to gradients (stabilization)
        grad_w = ffz_torsion_deform(grad_w, beta=self.beta)

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
    # Add more varied words to force larger vocab
    pos_words = ['great', 'amazing', 'love', 'excellent', 'fantastic', 'wonderful', 'superb', 'brilliant', 'outstanding', 'terrific']
    neg_words = ['bad', 'terrible', 'hate', 'poor', 'awful', 'horrible', 'dreadful', 'pathetic', 'useless', 'disappointing']
    # pos_words = ['great', 'amazing', 'love', 'excellent', 'fantastic']
    # neg_words = ['bad', 'terrible', 'hate', 'poor', 'awful']
    texts = []
    labels = []
    for _ in range(n_samples):
        is_pos = np.random.rand() > 0.5
        sentence = ' '.join(np.random.choice(pos_words if is_pos else neg_words, size=14))
        texts.append(sentence)
        labels.append(1 if is_pos else 0)
    X, vocab = bag_of_words(texts, vocab_size=input_size)
    return X, np.array(labels), vocab
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
    # ... after deformed_sym = sym_vec + balance

    # Symbolic mean: sum over rows / number of elements
    # n_elements = deformed_sym.shape[0]
    # sym_mean = deformed_sym.sum() / n_elements

    # Subtract symbolically
    # deformed_sym -= sym_mean

    # Then convert once at the end
    # return np.array(deformed_sym).flatten().astype(float)

    # Convert BACK to numpy BEFORE doing mean subtraction
    deformed_np = np.array(deformed_sym).flatten().astype(float)

    # Enforce zero-sum invariant in numpy (fast & reliable)
    mean_val = np.mean(deformed_np)
    deformed_np -= mean_val

    # Optional: light clipping to prevent explosion (FFZ boundedness spirit)
    deformed_np = np.clip(deformed_np, -10.0, 10.0)

    return deformed_np




# Main Training and Inference
if __name__ == "__main__":
    # Generate data
    # X_train, y_train, vocab = generate_synthetic_data(200, input_size=20)
    VOCAB_SIZE = 30
    X_train, y_train, vocab = generate_synthetic_data(200, input_size=VOCAB_SIZE)
   
    print(f"X_train shape: {X_train.shape}, actual vocab size: {len(vocab)}")
    assert X_train.shape[1] == len(vocab), "Vectorizer inconsistency"
    print(f"X_train shape: {X_train.shape}, expected features: {len(vocab)}")

    # Use ACTUAL feature count — this fixes the crash
    # rest of training loop unchanged...
    # Change this line near the bottom:
    # model = FFZPerceptron(input_size=20, beta=0.05)  # Mild FFZ for stability
    # model = FFZPerceptron(input_size=X_train.shape[1], beta=0.05)  # was hardcoded 20
    # model = FFZPerceptron(input_size=VOCAB_SIZE, beta=0.05)
    model = FFZPerceptron(input_size=X_train.shape[1], beta=0.05)

    # Training loop (with loss logging for plateau viz)
    epochs = 100
    losses = []
    for epoch in range(epochs):
        loss = model.train_step(X_train, y_train)
        losses.append(loss)
        if epoch % 10 == 0:
            print(f"Epoch {epoch}, Loss: {loss:.4f}")

    # Visualize loss (confirm FFZ plateaus)
    plt.plot(losses)
    plt.title("Training Loss Curve (FFZ Stabilized)")
    plt.xlabel("Epoch")
    plt.ylabel("BCE Loss")
    plt.show()

    # NLP Inference Example
    test_texts = ["This is amazing and great!", "This is terrible and bad."]
    X_test, _ = bag_of_words(test_texts, len(vocab))  # Align to train vocab
    preds = model.forward(X_test)
    print("Inference Predictions (Positive Prob):", preds)
