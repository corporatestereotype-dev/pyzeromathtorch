import numpy as np
import json
import argparse
import sys
from pathlib import Path
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

# ── FFZ-Stabilized Neural Network ────────────────────────────────────────────
class FFZPerceptron:
    def __init__(self, input_size, hidden_size=16, learning_rate=0.9, beta=0.0001):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.W1 = np.random.randn(input_size, hidden_size) * 0.01
        self.b1 = np.zeros(hidden_size)
        self.W2 = np.random.randn(hidden_size, 1) * 0.01
        self.b2 = 0.0
        self.lr = learning_rate
        self.beta = beta
        self.history = {'grad_norms': []}  # Track gradient statistics

    def forward(self, x):
        self.z1 = np.dot(x, self.W1) + self.b1
        self.a1 = 1 / (1 + np.exp(-self.z1))
        z2 = np.dot(self.a1, self.W2) + self.b2
        return 1 / (1 + np.exp(-z2))

    def loss(self, y_pred, y_true):
        return -np.mean(y_true * np.log(y_pred + 1e-5) + (1 - y_true) * np.log(1 - y_pred + 1e-5))

    def train_step(self, x, y, epoch=0, verbose=False):
        y_pred = self.forward(x)
        
        if y.ndim == 1:
            y = y.reshape(-1, 1)

        # Backprop
        error_out = y_pred - y
        grad_W2 = np.dot(self.a1.T, error_out) / len(y)
        grad_b2 = np.mean(error_out)

        error_hidden = np.dot(error_out, self.W2.T) * self.a1 * (1 - self.a1)
        grad_W1 = np.dot(x.T, error_hidden) / len(y)
        grad_b1 = np.mean(error_hidden, axis=0)

        # Log gradient norms before FFZ
        pre_norm_W1 = np.linalg.norm(grad_W1)
        pre_norm_W2 = np.linalg.norm(grad_W2)

        # Apply FFZ
        grad_W1 = ffz_torsion_deform(grad_W1.flatten(), beta=self.beta).reshape(grad_W1.shape)
        grad_W2 = ffz_torsion_deform(grad_W2.flatten(), beta=self.beta).reshape(grad_W2.shape)

        # Log after FFZ
        post_norm_W1 = np.linalg.norm(grad_W1)
        post_norm_W2 = np.linalg.norm(grad_W2)
        
        if verbose and epoch % 20 == 0:
            print(f"  Grad W1: {pre_norm_W1:.4f} → {post_norm_W1:.4f} (Δ={post_norm_W1-pre_norm_W1:.4f})")
            print(f"  Grad W2: {pre_norm_W2:.4f} → {post_norm_W2:.4f} (Δ={post_norm_W2-pre_norm_W2:.4f})")

        self.history['grad_norms'].append({
            'epoch': epoch,
            'W1_pre': pre_norm_W1, 'W1_post': post_norm_W1,
            'W2_pre': pre_norm_W2, 'W2_post': post_norm_W2
        })

        if not np.all(np.isfinite(grad_W1)) or not np.all(np.isfinite(grad_W2)):
            print("Warning: Non-finite gradients — clipping")
            grad_W1 = np.nan_to_num(grad_W1)
            grad_W2 = np.nan_to_num(grad_W2)

        # Update
        self.W1 -= self.lr * grad_W1
        self.b1 -= self.lr * grad_b1
        self.W2 -= self.lr * grad_W2
        self.b2 -= self.lr * grad_b2

        return self.loss(y_pred, y)

    def save(self, path_prefix="ffz_model"):
        """Save model weights and config"""
        np.save(f"{path_prefix}_W1.npy", self.W1)
        np.save(f"{path_prefix}_W2.npy", self.W2)
        np.save(f"{path_prefix}_b1.npy", self.b1)
        np.save(f"{path_prefix}_b2.npy", self.b2)
        
        config = {
            'input_size': self.input_size,
            'hidden_size': self.hidden_size,
            'learning_rate': self.lr,
            'beta': self.beta
        }
        with open(f"{path_prefix}_config.json", "w") as f:
            json.dump(config, f)
        print(f"Model saved to {path_prefix}_*.npy/json")

    @classmethod
    def load(cls, path_prefix="ffz_model"):
        """Load model from saved weights"""
        with open(f"{path_prefix}_config.json", "r") as f:
            config = json.load(f)
        
        model = cls(**config)
        model.W1 = np.load(f"{path_prefix}_W1.npy")
        model.W2 = np.load(f"{path_prefix}_W2.npy")
        model.b1 = np.load(f"{path_prefix}_b1.npy")
        model.b2 = np.load(f"{path_prefix}_b2.npy")
        print(f"Model loaded from {path_prefix}_*.npy")
        return model

# ── Synthetic Data Generator ────────────────────────────────────────────────
def generate_synthetic_data(n_samples=200, input_size=30):
    pos_words = ['great', 'amazing', 'love', 'excellent', 'fantastic', 'wonderful', 
                 'superb', 'brilliant', 'outstanding', 'terrific']
    neg_words = ['bad', 'terrible', 'hate', 'poor', 'awful', 'horrible', 
                 'dreadful', 'pathetic', 'useless', 'disappointing']
    texts, labels = [], []
    for _ in range(n_samples):
        is_pos = np.random.rand() > 0.5
        base_words = pos_words if is_pos else neg_words
        sentence_words = np.random.choice(base_words, size=25)
        if np.random.rand() > 0.6:
            key_word = np.random.choice(base_words)
            repeats = np.random.randint(2, 6)
            sentence_words = np.append(sentence_words, [key_word] * repeats)
        texts.append(' '.join(sentence_words))
        labels.append(1 if is_pos else 0)
    X, vocab = bag_of_words(texts, vocab_size=input_size)
    return X, np.array(labels), vocab

# ── Visualization ─────────────────────────────────────────────────────────────
def plot_training(losses, grad_norms, filename="ffz_training"):
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10))
    
    # Loss curve
    ax1.plot(losses, label='Training Loss', color='#1f77b4', linewidth=2)
    ax1.set_title("FFZ Torsion-Stabilized Training", fontsize=14, fontweight='bold')
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("BCE Loss")
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    
    # Gradient norms
    if grad_norms:
        epochs = [g['epoch'] for g in grad_norms if g['epoch'] % 10 == 0]
        w1_pre = [g['W1_pre'] for g in grad_norms if g['epoch'] % 10 == 0]
        w1_post = [g['W1_post'] for g in grad_norms if g['epoch'] % 10 == 0]
        
        ax2.plot(epochs, w1_pre, 'r--', alpha=0.5, label='W1 pre-FFZ')
        ax2.plot(epochs, w1_post, 'g-', label='W1 post-FFZ')
        ax2.set_title("Gradient Norm Stabilization", fontsize=12)
        ax2.set_xlabel("Epoch")
        ax2.set_ylabel("L2 Norm")
        ax2.legend()
        ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f"{filename}.png", dpi=300, bbox_inches='tight')
    plt.savefig(f"{filename}.pdf", bbox_inches='tight')
    print(f"Saved {filename}.png/.pdf")

# ── Interactive Mode ───────────────────────────────────────────────────────────
def interactive_mode(model, vocab):
    print("\n" + "="*70)
    print("🤖 INTERACTIVE MODE")
    print("Commands: quit | vocab | threshold <val> | save <name>")
    print("="*70)
    
    threshold = 0.5
    
    while True:
        try:
            user_input = input("\n> ").strip()
            if not user_input:
                continue
                
            cmd = user_input.lower()
            if cmd in ['quit', 'exit', 'q']:
                break
            elif cmd == 'vocab':
                print(f"\nVocabulary ({len(vocab)} words):")
                print(", ".join(vocab))
                continue
            elif cmd.startswith('threshold '):
                threshold = float(cmd.split()[1])
                print(f"Threshold set to {threshold}")
                continue
            elif cmd.startswith('save '):
                model.save(cmd.split()[1])
                continue
            
            # Predict
            X = vectorize_with_vocab([user_input], vocab)
            prob = model.forward(X).flatten()[0]
            
            sentiment = "POSITIVE" if prob >= threshold else "NEGATIVE"
            confidence = prob if prob >= 0.5 else 1-prob
            
            # Visual bar
            bar_len = 40
            fill = int(prob * bar_len)
            bar = "█" * fill + "░" * (bar_len - fill)
            
            print(f"\n  [{bar}] {prob:.4f}")
            print(f"  Sentiment: {sentiment} ({confidence:.1%} confidence)")
            
            # Coverage check
            words = set(re.findall(r'\w+', user_input.lower()))
            known = words & set(vocab)
            if len(words) > len(known):
                unknown = words - set(vocab)
                print(f"  ⚠️  Unknown: {', '.join(list(unknown)[:3])}")
                
        except KeyboardInterrupt:
            break
        except Exception as e:
            print(f"Error: {e}")

# ── Main ─────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description='FFZ Torsion-Stabilized Sentiment Classifier')
    parser.add_argument('--train', action='store_true', help='Train new model')
    parser.add_argument('--infer', action='store_true', help='Load saved model for inference')
    parser.add_argument('--epochs', type=int, default=500)
    parser.add_argument('--beta', type=float, default=0.05)
    parser.add_argument('--patience', type=int, default=50, help='Early stopping patience')
    parser.add_argument('--vocab-size', type=int, default=30)
    parser.add_argument('--model-name', default='ffz_model')
    args = parser.parse_args()

    if args.infer:
        # Inference only
        print(f"Loading model: {args.model_name}")
        model = FFZPerceptron.load(args.model_name)
        with open(f"{args.model_name}_vocab.json", "r") as f:
            vocab = json.load(f)
        interactive_mode(model, vocab)
        return

    # Training mode
    print(f"Generating synthetic data (vocab={args.vocab_size})...")
    X_train, y_train, vocab = generate_synthetic_data(200, input_size=args.vocab_size)
    
    print(f"Training: {X_train.shape}, Vocab: {len(vocab)}")
    model = FFZPerceptron(input_size=X_train.shape[1], hidden_size=16, 
                          learning_rate=0.01, beta=args.beta)

    # Training with early stopping
    losses = []
    best_loss = float('inf')
    patience_counter = 0
    min_delta = 1e-5
    
    print(f"\nTraining with early stopping (patience={args.patience})...")
    
    for epoch in range(args.epochs):
        loss = model.train_step(X_train, y_train, epoch=epoch, verbose=(epoch % 50 == 0))
        losses.append(loss)
        
        if epoch % 10 == 0:
            print(f"Epoch {epoch:3d}: Loss = {loss:.4f}")
        
        # Early stopping check
        if loss < best_loss - min_delta:
            best_loss = loss
            patience_counter = 0
        else:
            patience_counter += 1
            
        if patience_counter >= args.patience:
            print(f"\n✋ Early stopping at epoch {epoch} (best loss: {best_loss:.4f})")
            break
    
    # Save everything
    model.save(args.model_name)
    with open(f"{args.model_name}_vocab.json", "w") as f:
        json.dump(vocab, f)
    
    # Plot
    plot_training(losses, model.history['grad_norms'], filename=f"{args.model_name}_training")
    
    # Test examples
    tests = [
        "This is amazing and great and wonderful!",
        "This is terrible and bad and horrible!"
    ]
    X_test = vectorize_with_vocab(tests, vocab)
    preds = model.forward(X_test).flatten()
    print(f"\nQuick test: pos={preds[0]:.4f}, neg={preds[1]:.4f}")
    
    # Interactive
    interactive_mode(model, vocab)

if __name__ == "__main__":
    main()
