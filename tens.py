# -*- coding: utf-8 -*-
"""
Created on Fri Jun  8 12:32:04 2018

@author: STUDENT
"""
x = 1
y = x + 9
print(y)
import tensorflow as tf
x = tf.constant(1,name='x')
y = tf.Variable(x+9,name='y')
print(y)