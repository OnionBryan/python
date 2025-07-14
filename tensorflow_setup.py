import tensorflow as tf

def setup_tensorflow() -> tf.Tensor:
    """Initialize TensorFlow and compute a simple matrix product."""
    tf.config.experimental.set_visible_devices([], 'GPU')  # force CPU for consistency
    tf.random.set_seed(0)
    a = tf.constant([[1.0, 2.0], [3.0, 4.0]], dtype=tf.float32)
    b = tf.constant([[5.0], [6.0]], dtype=tf.float32)
    return tf.matmul(a, b)

if __name__ == "__main__":
    result = setup_tensorflow()
    print(result.numpy())
