import numpy as np;
import tensorflow as tf;
import DataLoader
import csv
import joblib
import random;
import sklearn.utils
import pandas as pd;
#from PIL import Image
import matplotlib.pyplot as plt;
import ModelLoader;

def showImg(V,txt):
    I = np.reshape(V,(28,28));
    plt.imshow(I,interpolation="nearest");
    plt.title("txt : " + txt);
    plt.show();


#np.random.seed(0);
#tf.set_random_seed(0);
#df = DataLoader.getPKLData();
df = DataLoader.LoadData();
X_train,X_test,Y_train,Y_test = DataLoader.splitData(df,0.05);
X_train = np.divide(X_train,256.);
X_test = np.divide(X_test,256.);
Y_train = Y_train.ravel();
Y_test = Y_test.ravel();

df = DataLoader.LoadData('../../data/DigitRecognizer/test.csv')
Xprod = np.array(df.values)
Xprod = np.transpose(Xprod);
Xprod = np.divide(Xprod,256.);

batches = ModelLoader.splitBatches(X_train,Y_train)
order = [i for i in range(len(batches))];

NK = 784
with tf.device('/gpu:0'):
    X = tf.placeholder(tf.float32,(NK,None),'X');
    Y = tf.placeholder(tf.int64, (None), 'Y');
    Y_ = tf.one_hot(Y,depth=10,axis=0);
    wDim = [NK,100,100,10];
    W = [];
    B = [];
    for i in range(1,len(wDim)):
        mag = np.sqrt(2.0/(wDim[i-1] + wDim[i]))
        W.append(tf.Variable(2*mag*np.random.rand(wDim[i-1],wDim[i]) - mag,dtype=tf.float32,name = "W" + str(i)));
        B.append(tf.Variable(np.zeros((wDim[i],1)),dtype=tf.float32,name="b"+str(i)));
    P = X;
    for i in range(len(W)):
        P = tf.matmul(tf.transpose(W[i]),P) + B[i];
        if i != len(W) - 1:
            P = tf.nn.relu(P);
        else:
            P = tf.nn.softmax(P,dim=0);

    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=Y_,logits=P,dim=0))*10;
    train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cost);
    #train_step = tf.train.AdamOptimizer(0.5).minimize(cost);

    tfpred = tf.argmax(P,dimension=0);



with tf.Session() as sess:
    saver = tf.train.Saver()
    init = tf.global_variables_initializer()
    sess.run(init);
    step = 3;
    lst = 0;
    loss = [];
    for i in range(100):
        cur_train = sess.run(cost, feed_dict={X: X_train, Y: Y_train});
        cur_test = sess.run(cost, feed_dict={X: X_test, Y: Y_test});
        if i % step == 0 :
            pred = sess.run(tfpred, feed_dict={X: X_train})
            tracc = sum(pred == Y_train) / len(Y_train) * 100;
            pred = sess.run(tfpred, feed_dict={X: X_test})
            teacc = sum(pred == Y_test) / len(Y_test) * 100;
            print("@iteration #%d" % (i + 1), "train loss =", cur_train, "test loss", cur_test, "train accuracy", tracc,
                  "test acc", teacc);
        loss.append(cur_train);
        random.shuffle(order)
        for k in order:
            cur_train = sess.run(cost, feed_dict={X: X_train, Y: Y_train});
            loss.append(cur_train);
            Xb,Yb = batches[k];
            sess.run(train_step,feed_dict={X:Xb,Y:Yb});
        if i and abs(cur_train - lst) < 1e-8: break;
        lst = cur_train;
    print ("final train loss",sess.run(cost,feed_dict={X:X_train,Y:Y_train}));
    print("final test loss", sess.run(cost, feed_dict={X:X_test, Y: Y_test}));
    pred = sess.run(tfpred,feed_dict={X:X_train})
    acc = sum(pred == Y_train)/len(Y_train)*100;
    print("train accuracy",acc)
    pred = sess.run(tfpred,feed_dict={X:X_test})
    for i in range(10):
        idx = i;
        showImg(X_test[:,idx],"pred = " + str(pred[idx]));
    plt.plot(loss);
    plt.show();
    pred = sess.run(tfpred, feed_dict={X: Xprod});
    f = open("out.csv","w");
    for i,label in enumerate(pred):
        f.write(str(i+1) + "," + str(label) + "\n");
    f.close();
    saver.save(sess,"batchs.ckpt");
    sess.close();


with tf.Session() as sess:
    saver = tf.train.Saver();
    saver.restore(sess,"batchs.ckpt");
    pred = sess.run(tfpred, feed_dict={X: Xprod});
    f = open("out.csv","w");
    f.write("ImageId,Label\n");
    for i,label in enumerate(pred):
        f.write(str(i+1) + "," + str(label) + "\n");
    f.close();
    sess.close();