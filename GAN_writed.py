#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jun  9 15:17:40 2018

@author: zhanglei
"""

import tensorflow as tf 
import gzip
import numpy as np
from scipy import misc
import logging
import time
import os

FLAGS=tf.flags.FLAGS


def initLogging(logFilename='record.log'):
  """Init for logging
  """
  logging.basicConfig(
                    level= logging.DEBUG,
                    format='%(asctime)s-%(levelname)s-%(message)s',
                    datefmt  = '%y-%m-%d %H:%M',
                    filename = logFilename,
                    filemode = 'w');
  console = logging.StreamHandler()
  console.setLevel(logging.INFO)
  formatter = logging.Formatter('%(asctime)s-%(levelname)s-%(message)s')
  console.setFormatter(formatter)
  logging.getLogger('').addHandler(console)
  
  
initLogging()


tf.flags.DEFINE_string('data_dir','data','the directory where the minist store in')
tf.flags.DEFINE_string('save_dir','result/','the val result stored in')
tf.flags.DEFINE_string('checkpoint','checkpoint/','the model stored in')
tf.flags.DEFINE_string('log','log','the summary result stored in')
tf.flags.DEFINE_integer('batch_size','64','the size of data batch')
tf.flags.DEFINE_integer('z_dim','62','the dim of noise')
tf.flags.DEFINE_integer('steps','1000000','the steps of training')
tf.flags.DEFINE_float('learning_rate','0.0002','learning rate')
tf.flags.DEFINE_float('beta1','0.5','beta')
tf.flags.DEFINE_bool('finetune','False','finetune or not')


#############prepare data#################

def extract_data(filename, num_data, head_size, data_size):
    with gzip.open(filename) as bytestream:
        bytestream.read(head_size)
        buf = bytestream.read(data_size * num_data)
        data = np.frombuffer(buf, dtype=np.uint8).astype(np.float)
    return data

def load_mnist(data_dir):
    data = extract_data(data_dir + '/train-images-idx3-ubyte.gz', 60000, 16, 28 * 28)
    trX = data.reshape((60000, 28, 28, 1))

    data = extract_data(data_dir + '/train-labels-idx1-ubyte.gz', 60000, 8, 1)
    trY = data.reshape((60000))

    data = extract_data(data_dir + '/t10k-images-idx3-ubyte.gz', 10000, 16, 28 * 28)
    teX = data.reshape((10000, 28, 28, 1))

    data = extract_data(data_dir + '/t10k-labels-idx1-ubyte.gz', 10000, 8, 1)
    teY = data.reshape((10000))

    trY = np.asarray(trY)
    teY = np.asarray(teY)

    X = np.concatenate((trX, teX), axis=0)
    y = np.concatenate((trY, teY), axis=0).astype(np.int)

    seed = 547
    np.random.seed(seed)
    np.random.shuffle(X)
    np.random.seed(seed)
    np.random.shuffle(y)

    y_vec = np.zeros((len(y), 10), dtype=np.float)
    for i, label in enumerate(y):
        y_vec[i, y[i]] = 1.0

    return X / 255., y_vec


X,y=load_mnist(FLAGS.data_dir)


###########get batch data################
def get_batch_data(data,batch_size,ep):
    length=X.shape[0]
    start=ep%length
    if start+batch_size<=length:
        return data[start:start+batch_size]
    else:
        return np.concatenate((data[start:],data[:start+batch_size-length]),axis=0)

    

##############net#######################

        
def conv(input_, shape,strides, name="conv2d"):
    with tf.variable_scope(name):
        w = tf.get_variable('w', shape=shape,initializer=tf.truncated_normal_initializer(stddev=0.02))
        conv = tf.nn.conv2d(input_, w, strides=strides, padding='SAME')

        biases = tf.get_variable('biases',shape[-1], initializer=tf.constant_initializer(0.0))
        conv = tf.reshape(tf.nn.bias_add(conv, biases), conv.get_shape())

        return conv
    
    

def lrelu(x):  
    return tf.maximum(x,x*0.2)
    



def batch_normalization(x, scope,train):
    return tf.contrib.layers.batch_norm(x,decay=0.9,updates_collections=None,epsilon=1e-5,scale=True,is_training=train,scope=scope)

    
def fc(input_, output_size, name=None, stddev=0.02, bias_start=0.0):
    shape = input_.get_shape().as_list()

    with tf.variable_scope(name or "Linear"):
        matrix = tf.get_variable("Matrix", [shape[1], output_size], tf.float32,tf.random_normal_initializer(stddev=stddev))
        bias = tf.get_variable("bias", [output_size],initializer=tf.constant_initializer(bias_start))

        return tf.matmul(input_, matrix) + bias    





def deconv(input_, shape, strides, outshape,name):
    with tf.variable_scope(name):
        # filter : [height, width, output_channels, in_channels]
        w = tf.get_variable('w', shape,initializer=tf.random_normal_initializer(stddev=0.02))


        deconv = tf.nn.conv2d_transpose(input_, w, output_shape=outshape, strides=strides)

        biases = tf.get_variable('biases', [outshape[-1]], initializer=tf.constant_initializer(0.0))
        deconv = tf.reshape(tf.nn.bias_add(deconv, biases), deconv.get_shape())

        
        return deconv

def discriminator(img_batch,train,reuse=False):
    with tf.variable_scope('discriminator',reuse=tf.AUTO_REUSE):
        conv_lr=lrelu(conv(img_batch,[4,4,1,64],[1,2,2,1],name='d_conv1'))
        conv_bn_lr=lrelu(batch_normalization(conv(conv_lr,[4,4,64,128],[1,2,2,1],name='d_conv2'),'d_bn1',train))
        reshaped=tf.reshape(conv_bn_lr,[FLAGS.batch_size,-1])
        fc_bn_lr=lrelu(batch_normalization(fc(reshaped,1024,name='d_fc1'),'d_bn2',train))
        out=tf.nn.sigmoid(fc(fc_bn_lr,1,name='d_fc2'))
        
    return out


def generator(noise,train,reuse):
    with tf.variable_scope('generator',reuse=tf.AUTO_REUSE):
        fc_bn_relu1=tf.nn.relu(batch_normalization(fc(noise,1024,name='g_fc1'),'g_bn1',train))
        fc_bn_relu2=tf.nn.relu(batch_normalization(fc(fc_bn_relu1,128 * 7 * 7,name='g_fc2'),'g_bn2',train))
        reshaped=tf.reshape(fc_bn_relu2,shape=[FLAGS.batch_size,7,7,128])
        de_bn_relu=tf.nn.relu(batch_normalization(deconv(reshaped,[4,4,64,128],[1,2,2,1],[FLAGS.batch_size,14,14,64],name='g_deconv1'),'g_bn3',train))
        out=tf.nn.sigmoid(deconv(de_bn_relu,[4,4,1,64],[1,2,2,1],[FLAGS.batch_size, 28, 28, 1],name='g_deconv2'))
        
    return out 





def save(img_arr,save_dir,ep):
    img_arr=np.squeeze(img_arr,-1)
    img=np.zeros((224,224))
    k=0
    for i in range(8):
        for j in range(8):
            img[i*28:i*28+28,j*28:j*28+28]=img_arr[k]
            k=k+1
    misc.imsave(FLAGS.save_dir+str(ep)+'.png',img)



def train():

    X_tr=tf.placeholder(dtype=tf.float32,shape=[FLAGS.batch_size,28,28,1],name='real_img')
    z=tf.placeholder(dtype=tf.float32,shape=[FLAGS.batch_size,FLAGS.z_dim],name='noise')
    
    d_real=discriminator(X_tr,True,False)
    
    g_out=generator(z,True,False)
    d_fake=discriminator(g_out,True,True)
    
    fake=generator(z,False,True)
    
    d_real_loss=-tf.reduce_mean(tf.log(d_real +1e-8 ))#################
    d_fake_loss=-tf.reduce_mean(tf.log(1 - d_fake  +1e-8))    
    d_loss=d_real_loss+d_fake_loss
   
    tf.summary.scalar('d_loss',d_loss)
    g_loss=-tf.reduce_mean(tf.log(d_fake  +1e-8))
    tf.summary.scalar('g_loss',g_loss)
    
        
    t_vars=tf.trainable_variables()
    v_d=[var for var in t_vars if 'd_'in var.name ]##################
    v_g=[var for var in t_vars if 'g_'in var.name ]
    
#    update_ops = tf.get_collection(tf.get_collection(tf.GraphKeys.UPDATE_OPS))
#    with tf.control_dependencies(update_ops):
    with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
#        print(d_real)
        d_op=tf.train.AdamOptimizer(learning_rate=FLAGS.learning_rate,beta1=FLAGS.beta1).minimize(d_loss,var_list=v_d)       
        g_op=tf.train.AdamOptimizer(learning_rate=5*FLAGS.learning_rate,beta1=FLAGS.beta1).minimize(g_loss,var_list=v_g) 
    
    
    with tf.Session() as sess:
        merged=tf.summary.merge_all()
        train_writer=tf.summary.FileWriter(logdir=FLAGS.log)
        saver=tf.train.Saver()
        
        if FLAGS.finetune==False:
            sess.run(tf.global_variables_initializer())
            sess.run(tf.local_variables_initializer())
        else:
            ckpt=tf.train.get_checkpoint_state(FLAGS.checkpoint)
            saver.restore(sess,ckpt.model_checkpoint_path)
            
        
        
        time_start=time.time()
        z_val=np.random.uniform(-1,1,size=[FLAGS.batch_size,FLAGS.z_dim])
        for ep in range(FLAGS.steps):
            data_batch=get_batch_data(X,FLAGS.batch_size,ep)
            noise=np.random.uniform(-1, 1,size=(FLAGS.batch_size,FLAGS.z_dim)).astype(np.float32)###################
            
            _,discriminator_loss=sess.run([d_op,d_loss],feed_dict={X_tr:data_batch,z:noise})
            _,generator_loss,summary=sess.run([g_op,g_loss,merged],feed_dict={X_tr:data_batch,z:noise})
            
            time_end=time.time()
            if ep%300==0 and ep!=0:
                print('\r|'+'>'*40+'|'+'%d/%d' %(300,300)+'  current d_loss is=%f,g_loss=%f,time_spended=%f ' %(discriminator_loss,generator_loss,time_end-time_start),end=' ')
                fake_img=sess.run([fake],feed_dict={z:z_val})##################
                fake_img=fake_img[0]
                save(fake_img,FLAGS.save_dir,ep)
                saver.save(sess,FLAGS.checkpoint,global_step=ep)
            else:
                trained_part=int((ep%300)/300*40)
                rest_part=40-int((ep%300)/300*40)
                print('\r|'+'>'*trained_part+' '*rest_part+'|'+'%d/%d' %(ep%300,300)+'  current d_loss is=%f,g_loss=%f,time_spended=%f ' %(discriminator_loss,generator_loss,time_end-time_start),end=' ')
            if ep%10==0:
                train_writer.add_summary(summary,ep)
                train_writer.flush()
        train_writer.close()

               
            
if __name__=='__main__':
    if not os.path.isdir(FLAGS.save_dir):
        os.makedirs(FLAGS.save_dir)
    if not os.path.isdir(FLAGS.checkpoint):
        os.makedirs(FLAGS.checkpoint)
    if not os.path.isdir(FLAGS.log):
        os.makedirs(FLAGS.log)
    train()  




        
