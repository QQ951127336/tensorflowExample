import tensorflow as tf
a = tf.constant([3.0,2.0])
b = tf.constant([3.0,4.0])
c = a * b
sess = tf.Session()
print(sess.run(c))
sess.close()

matrix1 = tf.constant([[3.,3.]])
matrix2 = tf.constant([[2.],[2.]])
product = tf.matmul(matrix1,matrix2)
with tf.Session() as sess:
    result = sess.run([product])
    print(result)

input1 = tf.constant(3.0)
input1 = tf.placeholder(tf.float32)
input2 = tf.placeholder(tf.float32)
output = tf.multiply(input1,input2)
with tf.Session() as sess:
    print(sess.run([output], feed_dict={input1:[7.],input2:[2.]}))
    