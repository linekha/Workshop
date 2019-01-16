!pip3 install https://storage.googleapis.com/tensorflow-builds/boromir/ubuntu-16.04/gpu-cuda-9.2-cudnn-7.1/tensorflow-1.8.0-cp36-cp36m-linux_x86_64.whl
import tensorflow as tf


from tensorflow.python.client import device_lib

device_lib.list_local_devices()

# Creates a graph.
a = tf.constant([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], shape=[2, 3], name='a')
b = tf.constant([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], shape=[3, 2], name='b')
c = tf.matmul(a, b)
# Creates a session with log_device_placement set to True.
sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))
# Runs the op.
print(sess.run(c))