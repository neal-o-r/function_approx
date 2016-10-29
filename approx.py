import tensorflow as tf
import numpy as np
import net as net
from tqdm import tqdm
import matplotlib.pyplot as plt
plt.style.use('fivethirtyeight')


n_example = 1000
n_batch   = 50
n_epoch   = 300
f_range   = (-5,5) 

np.random.seed(123) 
fx_true = lambda x: x**2 + x - 10
fx = lambda x: fx_true(x) + 0.1*np.random.randn(*x.shape)

def make_plot(train, guess):

        xs = np.linspace(f_range[0], f_range[1], 100)

        plt.plot(train[0], train[1], 'o', label='Training Data')
        plt.plot(xs, fx_true(xs), label='True Function')
        plt.plot(test[0], guesses, 'o', label='Test Data')
        plt.legend(loc='best')
        plt.xlim(1.2*f_range[0], 1.2*f_range[1])
        plt.ylim(1.2*min(train[1])[0], 1.2*max(train[1])[0])

        plt.show()



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
        
        l0 = tf.nn.softplus(net.feed_forward(x, 20, 'approx', 'l0'))
        l1 = net.feed_forward(l0, 1, 'approx', 'l1')
            
        return l1


train, test = get_data(fx, n_example)

X = tf.placeholder(tf.float32, [1, 1], name="X")
Y = tf.placeholder(tf.float32, [1, 1], name="Y")

y_c = approximator(X)

cost = tf.nn.l2_loss(y_c - Y)
train_op = tf.train.GradientDescentOptimizer(0.005).minimize(cost)

guesses = []
with tf.Session() as sess:
        
        sess.run(tf.initialize_all_variables())

        for i in range(n_epoch):
        
                inds = np.random.random_integers(0, 
                        len(train[0])-1, size=n_batch) 

                x_b = train[0].ravel()[inds]
                y_b = train[1].ravel()[inds]               
        
                for x, y in zip(x_b, y_b):
                        
                        sess.run(train_op, feed_dict={X: np.array([[x]]), 
                                                      Y: np.array([[y]])})

                if i%100 == 0 :
                        mse = 0.
                        for x, y in zip(test[0].ravel(), test[1].ravel()):

                                mse += sess.run(cost, feed_dict={X:np.array([[x]]),
                                                                 Y:np.array([[y]])})
                        

                        print("Epoch %i, MSE %g" %(i, mse))
        
        for x, y in zip(test[0].ravel(), test[1].ravel()):

                guesses.append(sess.run(y_c,  feed_dict={X:np.array([[x]])})[0][0])
                        
make_plot(train, guesses)
