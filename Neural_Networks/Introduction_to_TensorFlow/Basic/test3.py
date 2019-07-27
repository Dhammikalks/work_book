import tensorflow as ts

node1  = ts.placeholder(ts.float32)
node2  = ts.placeholder(ts.float32)

node3 = node1+node2


with ts.Session() as sess:
    file_writer = ts.summary.FileWriter('/home/ros/Desktop/work_book/Neural_Networks/Introduction_to_TensorFlow/Basic/graph',sess.graph)
    output = sess.run(node3,{node1: [1,2],node2: [3,4]})
    print(output)
