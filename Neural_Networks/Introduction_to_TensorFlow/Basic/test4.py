import  tensorflow as tf

#Model parameters

w = tf.Variable([.3],tf.float32)
b = tf.Variable([-.3],tf.float32)


#Input and outputs
x = tf.placeholder(tf.float32)

linear_model = w * x + b

y = tf.placeholder(tf.float32)

sequre_delta = tf.square(y - linear_model)
loss = tf.reduce_sum(sequre_delta)


optimizer = tf.train.GradientDescentOptimizer(0.01)
train = optimizer.minimize(loss)

init = tf.global_variables_initializer()


with tf.Session() as sess:
    file_writer = tf.summary.FileWriter('/home/ros/Desktop/work_book/Neural_Networks/Introduction_to_TensorFlow/Basic/graph',sess.graph)
    sess.run(init)
    for i in range(1000):
        sess.run(train,{x: [1,2,3,4], y:[-0,-1,-2,-3]})
    print(sess.run([w,b]))
