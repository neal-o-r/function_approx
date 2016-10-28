import tensorflow as tf
import numpy as np
import net as net
from tqdm import tqdm
import matplotlib.pyplot as plt
plt.style.use('fivethirtyeight')


n_example = 1000
n_batch   = 50
n_epoch   = 300

np.random.seed(123) 
fx = lambda x: x + 0.1*np.random.randn(*x.shape)

def get_data(fx, n, x_range=[-5,5]):

        all_x = np.float32(
              np.random.uniform(x_range[0], x_range[1], (1, n))).T
        
        np.random.shuffle(all_x)
        
        train_x = all_x[:9 *(n // 10)]
        test_x  = all_x[9 *(n // 10):]
        train_y = fx(train_x)
        test_y  = fx(test_x)

        return (train_x, train_y), (test_x, test_y)


def approximator(x):
        # takes an x value and returns a y
        l0 = tf.nn.softplus(net.feed_forward(x, x.get_shape()[1], 'gen0'))
        l1 = net.feed_forward(l0, l0.get_shape()[1], 'gen1')
 
        return l1

train, test = get_data(fx, n_example)

X = tf.placeholder(tf.float32, [None, 1], name="X")
Y = tf.placeholder(tf.float32, [None, 1], name="Y")

y_c = approximator(X)

cost = tf.nn.l2_loss(y_c - Y)
train_op = tf.train.GradientDescentOptimizer(0.05).minimize(cost)

guesses = []
with tf.Session() as sess:
        
        sess.run(tf.initialize_all_variables())

        for i in range(n_epoch):

                x_b = np.random.choice(train[0].ravel(), size=n_batch)
                y_b = np.random.choice(train[1].ravel(), size=n_batch)                

                for x, y in zip(x_b, y_b):
                        
                        sess.run(train_op, feed_dict={X: np.array([[x]]), 
                                                      Y: np.array([[y]])})
                if i%100 == 0 :
                        mse = 0.
                        for x, y in zip(test[0].ravel(), test[1].ravel()):

                                mse += sess.run(tf.nn.l2_loss(y_c - y),  feed_dict={X:np.array([[x]])})
                        print("Epoch %i, MSE %g" %(i, mse))
        
        for x, y in zip(test[0].ravel(), test[1].ravel()):

                guesses.append(sess.run(y_c,  feed_dict={X:np.array([[x]])})[0][0])
                        
plt.plot(train[0], train[1], 'o') 
plt.plot(test[0], guesses, 'o')
plt.show()
