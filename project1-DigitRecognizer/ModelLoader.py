import numpy as np;
import tensorflow as tf;
import sklearn

# NK = 784
#
# with tf.device('/gpu:0'):
#     X = tf.placeholder(tf.float64,(NK,None),'X');
#     Y = tf.placeholder(tf.float64, (1, None), 'Y');
#     wDim = [NK,4,8,4,1];
#     W = [];
#     B = [];
#     for i in range(1,len(wDim)):
#         mag = np.sqrt(2.0/(wDim[i-1] + wDim[i]));
#         W.append(tf.Variable(mag*np.random.rand(wDim[i-1],wDim[i]),dtype=tf.float64,name = "W" + str(i)));
#         B.append(tf.Variable(np.zeros((wDim[i],1)),dtype=tf.float64,name="b"+str(i)));
#
#     P = X;
#     for i in range(len(W)):
#         P = tf.matmul(tf.transpose(W[i]),P) + B[i];
#         if i != len(W) - 1: P = tf.nn.relu(P);
#
#     cost = tf.reduce_mean(tf.square(Y - P,name="cost"));
#     train_step = tf.train.GradientDescentOptimizer(0.1).minimize(cost);
#
#
# with tf.Session() as sess:
#     init = tf.global_variables_initializer()
#     sess.run(init);
#     for i in range(len(W)):
#         sess.run(W[i]);
#         sess.run(B[i]);
#     step = 5;
#     lst = 0;
#     for i in range(301):
#         cur = sess.run(cost, feed_dict={X: iX, Y: iY});
#         if i%step == 0: print("@iteration #%d"%(i+1),"accuracy =",cur);
#         sess.run(train_step,feed_dict={X:iX,Y:iY});
#         if i and abs(cur - lst) < 1e-8: break;
#         lst = cur;
#     print ("final train accuracy",sess.run(cost,feed_dict={X:iX,Y:iY}));
#     print("final test accuracy", sess.run(cost, feed_dict={X: iX, Y: iY}));

def splitBatches(X,Y,batch_size = 2000):
    ret = [];
    m = len(Y);
    s = 0;
    while s < m:
        e = min(m,s + batch_size);
        ret.append([X[:,s:e],Y[s:e]]);
        s = e
    return ret;