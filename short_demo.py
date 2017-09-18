from __future__ import print_function,division
import numpy as np
import tensorflow as tf

#create a Variable
w=tf.Variable(initial_value=[[1,2],[3,4]],dtype=tf.float32)
x=tf.Variable(initial_value=[[1,1],[1,1]],dtype=tf.float32)
y=tf.matmul(w,x)
z=tf.sigmoid(y)
print(z)
init_op=tf.global_variables_initializer()

with tf.Session() as session:
    session.run(init_op)
    z=session.run(z)
    print(z)