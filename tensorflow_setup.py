import tensorflow as tf


def setup_tensorflow():
    """Initialize TensorFlow and report available devices."""
    gpus = tf.config.list_physical_devices('GPU')
    print(f"TensorFlow version: {tf.__version__}")
    if gpus:
        print(f"GPUs detected: {len(gpus)}")
    else:
        print("No GPU detected. Using CPU.")
    return gpus


def demo_matrix_operations():
    """Run a small matrix multiplication demo."""
    a = tf.constant([[1.0, 2.0], [3.0, 4.0]])
    b = tf.constant([[5.0, 6.0], [7.0, 8.0]])
    c = tf.linalg.matmul(a, b)
    print("Matrix A:\n", a.numpy())
    print("Matrix B:\n", b.numpy())
    print("A @ B:\n", c.numpy())


if __name__ == "__main__":
    setup_tensorflow()
    demo_matrix_operations()
