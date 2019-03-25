import tensorflow as tf
import numpy as np
import sys
import csv
import json
from data_pre import *
import vision
import copy
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from lifelines.utils import concordance_index
from supersmoother import SuperSmoother
import warnings
warnings.filterwarnings('ignore')
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"]='3'

class snn(object):
    def __init__(self, input_node, hidden_layers_node, output_node,
                 learning_rate=0.001, learning_rate_decay=1.0, activation='tanh',
                 L2_reg=0.0, L1_reg=0.0, optimizer='sgd', dropout_keep_prob=1.0, ties='noties'):

        self.train_data = dict()
        self.train_data['ties'] = ties
        self.configuration = {
            'input_node': input_node,
            'hidden_layers_node': hidden_layers_node,
            'output_node': output_node,
            'learning_rate': learning_rate,
            'learning_rate_decay': learning_rate_decay,
            'activation': activation,
            'L1_reg': L1_reg,
            'L2_reg': L2_reg,
            'optimizer': optimizer,
            'dropout': dropout_keep_prob
        }

    def load_data(self,X,label):
        self.train_data['X'], self.train_data['E'], \
            self.train_data['T'], self.train_data['failures'], \
            self.train_data['atrisk'], self.train_data['ties'] = parse_data(X, label)
        # print('pre success')

    def generate_graph(self,seed=1):
        G = tf.Graph()
        with G.as_default():
            tf.set_random_seed(seed)

            X = tf.placeholder(tf.float32, [None, self.configuration['input_node']], name = 'x-input')
            y_ = tf.placeholder(tf.float32, [None, self.configuration['output_node']], name = 'label-input')

            keep_prob = tf.placeholder(tf.float32, name = 'keep_prob')

            self.nnweights = []
            self.nnbias = []
            prev_node = self.configuration['input_node']
            prev_x = X
            for i in range(len(self.configuration['hidden_layers_node'])):
                layer_name = 'layer' + str(i+1)
                with tf.variable_scope(layer_name, reuse=tf.AUTO_REUSE):
                    weights = tf.get_variable('weights',[prev_node,self.configuration['hidden_layers_node'][i]],initializer=tf.truncated_normal_initializer(stddev=0.1))
                    self.nnweights.append(weights)

                    biases = tf.get_variable('biases',[self.configuration['hidden_layers_node'][i]],initializer=tf.constant_initializer(0.0))
                    self.nnbias.append(biases)
                    layer_out = tf.nn.dropout(tf.matmul(prev_x,weights)+biases, keep_prob)

                    if self.configuration['activation'] == 'relu':
                        layer_out = tf.nn.relu(layer_out)
                    elif self.configuration['activation'] == 'sigmoid':
                        layer_out = tf.nn.sigmoid(layer_out)
                    elif self.configuration['activation'] == 'tanh':
                        layer_out = tf.nn.tanh(layer_out)
                    else:
                        raise NotImplementedError('activation not recognized')

                    prev_node = self.configuration['hidden_layers_node'][i]
                    prev_x = layer_out

            layer_name = 'layer_last'
            with tf.variable_scope(layer_name, reuse=tf.AUTO_REUSE):
                weights = tf.get_variable('weights',[prev_node,self.configuration['output_node']],initializer=tf.truncated_normal_initializer(stddev=0.1))
                self.nnweights.append(weights)
                biases = tf.get_variable('biases',[self.configuration['output_node']],initializer=tf.constant_initializer(0.0))
                self.nnbias.append(biases)

                layer_out = tf.matmul(prev_x,weights)+biases

            y = layer_out
            # Global step
            with tf.variable_scope('training_step', reuse=tf.AUTO_REUSE):
                global_step = tf.get_variable(
                    "global_step",
                    [],
                    dtype=tf.int32,
                    initializer=tf.constant_initializer(0),
                    trainable=False
                )
            # Loss value
            reg_item = tf.contrib.layers.l1_l2_regularizer(self.configuration['L1_reg'], self.configuration['L2_reg'])
            reg_term = tf.contrib.layers.apply_regularization(reg_item, self.nnweights)
            loss_fun = self._negative_log_likelihood(y_, y)
            loss = loss_fun + reg_term

            if self.configuration['optimizer'] == 'sgd':
                lr = tf.train.exponential_decay(self.configuration['learning_rate'],global_step,1,self.configuration['learning_rate_decay'])
                train_step = tf.train.GradientDescentOptimizer(lr).minimize(loss, global_step=global_step)
            elif self.configuration['optimizer'] == 'adam':
                train_step = tf.train.AdamOptimizer(self.configuration['learning_rate']).minimize(loss, global_step=global_step)
            else:
                raise NotImplementedError('optimization not recognized')

            init_op = tf.global_variables_initializer()
            saver = tf.train.Saver(max_to_keep=1)

        # print('graph success')
        self.X = X
        self.y_ = y_
        self.keep_prob = keep_prob
        self.y = y
        self.global_step = global_step
        self.loss = loss
        self.train_step = train_step

        self.saver = saver
        self.sess = tf.Session(graph=G)
        self.sess.run(init_op)

    def load_model(self):

        G = tf.Graph()
        with G.as_default():
            tf.set_random_seed(1)
            X = tf.placeholder(tf.float32, [None, self.configuration['input_node']], name = 'x-input')
            y_ = tf.placeholder(tf.float32, [None, self.configuration['output_node']], name = 'label-input')
            keep_prob = tf.placeholder(tf.float32, name = 'keep_prob')

            self.nnweights = []
            self.nnbias = []
            prev_node = self.configuration['input_node']
            prev_x = X
            for i in range(len(self.configuration['hidden_layers_node'])):
                layer_name = 'layer' + str(i+1)
                with tf.variable_scope(layer_name, reuse=tf.AUTO_REUSE):
                    weights = tf.get_variable('weights',[prev_node,self.configuration['hidden_layers_node'][i]],initializer=tf.truncated_normal_initializer(stddev=0.1))
                    self.nnweights.append(weights)

                    biases = tf.get_variable('biases',[self.configuration['hidden_layers_node'][i]],initializer=tf.constant_initializer(0.0))
                    self.nnbias.append(biases)
                    layer_out = tf.nn.dropout(tf.matmul(prev_x,weights)+biases, keep_prob)

                    if self.configuration['activation'] == 'relu':
                        layer_out = tf.nn.relu(layer_out)
                    elif self.configuration['activation'] == 'sigmoid':
                        layer_out = tf.nn.sigmoid(layer_out)
                    elif self.configuration['activation'] == 'tanh':
                        layer_out = tf.nn.tanh(layer_out)
                    else:
                        raise NotImplementedError('activation not recognized')

                    prev_node = self.configuration['hidden_layers_node'][i]
                    prev_x = layer_out

            layer_name = 'layer_last'
            with tf.variable_scope(layer_name, reuse=tf.AUTO_REUSE):
                weights = tf.get_variable('weights',[prev_node,self.configuration['output_node']],initializer=tf.truncated_normal_initializer(stddev=0.1))
                self.nnweights.append(weights)
                biases = tf.get_variable('biases',[self.configuration['output_node']],initializer=tf.constant_initializer(0.0))
                self.nnbias.append(biases)

                layer_out = tf.matmul(prev_x,weights)+biases

            y = layer_out

            saver = tf.train.Saver()

        # print('graph success')
        self.X = X
        self.y_ = y_
        self.keep_prob = keep_prob
        self.y = y

        self.sess = tf.Session(graph=G)
        saver.restore(self.sess,"survival/model/model.ckpt")

    def train(self, X_test, y_test, num_epoch = 5000, iteration=-1):
        loss_list = []
        CI_list = []
        N = self.train_data['E'].shape[0]
        best_CI = 0
        for i in range(num_epoch):
            _, output_y, loss_value, step = self.sess.run(
                    [self.train_step, self.y, self.loss, self.global_step],
                    feed_dict= {
                        self.X: self.train_data['X'],
                        self.y_: self.train_data['E'].reshape((N,1)),
                        self.keep_prob: self.configuration['dropout']
                    }
                )
            loss_list.append(loss_value)
            label = {
                't': self.train_data['T'],
                'e': self.train_data['E']
            }
            CI = self._metrics_ci(label, output_y)
            CI_list.append(CI)

            valid_CI = self.score(X_test, y_test)
            # T0, ST = model.survival_function(np.array(data),algo="wwe")
            # count = 0
            # for i in range(len(model.train_data['E'])):
            #     if (ST[i,np.where(T0==model.train_data['T'][i])] >= 0.5 and model.train_data['E'][i]>0) or (ST[i,np.where(T0==model.train_data['T'][i])] < 0.5 and model.train_data['E'][i]==0):
            #         count += 1
            # cur_value = count*1.0/len(model.train_data['E'])
            if (iteration != -1) and (i % iteration == 0):
                # print("-------------------------------------------------")
                # print("training steps %d:\nloss = %g.\n" % (step, loss_value))
                # print("CI = %g.\n" % CI)
                # print("vaild_CI = %g. \n" % valid_CI)
                if valid_CI > best_CI:
                    best_CI = valid_CI
                self.saver.save(self.sess, "model/model.ckpt")

    def _negative_log_likelihood(self, y_true, y_pred):
        logL = 0
        # pre-calculate cumsum
        cumsum_y_pred = tf.cumsum(y_pred)
        hazard_ratio = tf.exp(y_pred)
        cumsum_hazard_ratio = tf.cumsum(hazard_ratio)
        if self.train_data['ties'] == 'noties':
            log_risk = tf.log(cumsum_hazard_ratio)
            likelihood = y_pred - log_risk
            # dimension for E: np.array -> [None, 1]
            uncensored_likelihood = likelihood * y_true
            logL = -tf.reduce_sum(uncensored_likelihood)
        else:
            # Loop for death times
            for t in self.train_data['failures']:
                tfail = self.train_data['failures'][t]
                trisk = self.train_data['atrisk'][t]
                d = len(tfail)
                logL += -cumsum_y_pred[tfail[-1]] + (0 if tfail[0] == 0 else cumsum_y_pred[tfail[0]-1])

                if self.train_data['ties'] == 'breslow':
                    s = cumsum_hazard_ratio[trisk[-1]]
                    logL += tf.log(s) * d
                elif self.train_data['ties'] == 'efron':
                    s = cumsum_hazard_ratio[trisk[-1]]
                    r = cumsum_hazard_ratio[tfail[-1]] - (0 if tfail[0] == 0 else cumsum_hazard_ratio[tfail[0]-1])
                    for j in range(d):
                        logL += tf.log(s - j * r / d)
                else:
                    raise NotImplementedError('tie breaking method not recognized')
        # negative average log-likelihood
        observations = tf.reduce_sum(y_true)
        return logL / observations

    def _metrics_ci(self, label_true, y_pred):
        hr_pred = -y_pred
        ci = concordance_index(label_true['t'], hr_pred, label_true['e'])
        return ci

    def predict(self, X):

        risk = self.sess.run([self.y],feed_dict={self.X: X, self.keep_prob: 1.0})
        risk = np.squeeze(risk)
        if risk.shape == ():
            risk = risk.reshape((1, ))
        return risk
    def score(self,X,label):
        risk = self.predict(X)
        return self._metrics_ci(label,risk)
    def close(self):
        self.sess.close()
        # print("session closed!")
    def get_vip_byweights(self):
        # Fetch weights of network
        W = [self.sess.run(w) for w in self.nnweights]
        n_w = len(W)
        # Matrix multiplication for all hidden layers except last output layer
        hiddenMM = W[- 2].T
        for i in range(n_w - 3, -1, -1):
            hiddenMM = np.dot(hiddenMM, W[i].T)
        # Multiply last layer matrix and compute the sum of each variable for VIP
        last_layer = W[-1]
        s = np.dot(np.diag(last_layer[:, 0]), hiddenMM)

        sumr = s / s.sum(axis=1).reshape(s.shape[0] ,1)
        score = sumr.sum(axis=0)
        VIP = score / score.max()
        # for i, v in enumerate(VIP):
            # print("%dth feature score : %g." % (i, v))
        return VIP

    def survival_function(self, X, algo="wwe", base_X=None, base_label=None,
                          smoothed=False, is_plot=True):

        risk = self.predict(X)
        hazard_ratio = np.exp(risk.reshape((risk.shape[0], 1)))
        # Estimate S0(t) using data(base_X, base_label)
        T0, S0 = self.base_surv(algo=algo, X=base_X, label=base_label, smoothed=smoothed)
        ST = S0**(hazard_ratio)
        # if is_plot:
        #     vision.plot_surv_func(T0, ST)
        return T0, ST

    def base_surv(self, algo="wwe", X=None, label=None, smoothed=False):

        if X is None or label is None:
            X = self.train_data['X']
            label = {'t': self.train_data['T'],
                     'e': self.train_data['E']}
        X, E, T, failures, atrisk, ties = parse_data(X, label)

        s0 = [1]
        t0 = [0]
        risk = self.predict(X)
        hz_ratio = np.exp(risk)
        if algo == 'wwe':
            for t in T[::-1]:
                if t in t0:
                    continue
                t0.append(t)
                if t in atrisk:
                    # R(t_i) - D_i
                    trisk = [j for j in atrisk[t] if j not in failures[t]]
                    dt = len(failures[t]) * 1.0
                    s = np.sum(hz_ratio[trisk])
                    cj = 1 - dt / (dt + s)
                    s0.append(cj)
                else:
                    s0.append(1)
        elif algo == 'kp':
            for t in T[::-1]:
                if t in t0:
                    continue
                t0.append(t)
                if t in atrisk:
                    # R(t_i)
                    trisk = atrisk[t]
                    s = np.sum(hz_ratio[trisk])
                    si = hz_ratio[failures[t][0]]
                    cj = (1 - si / s) ** (1 / si)
                    s0.append(cj)
                else:
                    s0.append(1)
        elif algo == 'bsl':
            for t in T[::-1]:
                if t in t0:
                    continue
                t0.append(t)
                if t in atrisk:
                    # R(t_i)
                    trisk = atrisk[t]
                    dt = len(failures[t]) * 1.0
                    s = np.sum(hz_ratio[trisk])
                    cj = 1 - dt / s
                    s0.append(cj)
                else:
                    s0.append(1)
        else:
            raise NotImplementedError('tie breaking method not recognized')
        # base survival function
        S0 = np.cumprod(s0, axis=0)
        T0 = np.array(t0)

        if smoothed:
            # smooth the baseline hazard
            ss = SuperSmoother()
            #Check duplication points
            ss.fit(T0, S0, dy=100)
            S0 = ss.predict(T0)

        return T0, S0

def cur(info):
#load data
  X_train = []
  train_e = []
  train_t = []
  # X_test = []
  # test_e = []
  # test_t = []
  with open('survival/data/train_LCCS_onehot.csv') as f:
      lines = csv.reader(f)
      head = next(lines)
      for line in lines:
          X_train.append(line)

  with open('survival/data/train_LCCS_label.csv') as f:
      lines = csv.reader(f)
      head = next(lines)
      for line in lines:
          train_e.append(int(line[0]))
          train_t.append(int(line[1]))
  #
  # with open('./data/test_LCCS_onehot.csv') as f:
  #     lines = csv.reader(f)
  #     head = next(lines)
  #     for line in lines:
  #         X_test.append(line)
  #
  # with open('./data/test_LCCS_label.csv') as f:
  #     lines = csv.reader(f)
  #     head = next(lines)
  #     for line in lines:
  #         test_e.append(int(line[0]))
  #         test_t.append(int(line[1]))
  #
  y_train = {'e':train_e, 't':train_t}
  # y_test = {'e': test_e, 't':test_t}
  #
  x_train = []
  x_test = []
  y_test_index = []
  for p in X_train:
      x_train.append(p[1:])
  # for p in X_test:
  #     y_test_index.append(p[0])
  #     x_test.append(p[1:])
  #
  # print(len(x_train[0]))
  # x_train = np.array(x_train)
  # f=[[],[],[],[]]
  # for j in range(len(x_train)):
  #   for i in range(4):
  #     f[i].append(int(x_train[j,i]))

  # print(len(x_test[0]))
  # print(np.array(x_train))
  # print(np.max(f,1))
  # print(np.min(f,1))
  scaler = MinMaxScaler().fit(x_train)
  x_train = scaler.transform(x_train)
  # print(x_train)
  # # x_test = scaler.transform(x_test)
  #
  # # x_test_final = []
  # # for d in x_test:
  # #     d1 = copy.deepcopy(d)
  # #     d1[61] = 1.0
  # #     d1[62] = 0.0
  # #     d1[63] = 0.0
  # #     d1[64] = 0.0
  # #     x_test_final.append(d1)
  # #     d2 = copy.deepcopy(d)
  # #     d2[61] = 0.0
  # #     d2[62] = 1.0
  # #     d2[63] = 0.0
  # #     d2[64] = 0.0
  # #     x_test_final.append(d2)
  # #     d3 = copy.deepcopy(d)
  # #     d3[61] = 0.0
  # #     d3[62] = 0.0
  # #     d3[63] = 1.0
  # #     d3[64] = 0.0
  # #     x_test_final.append(d3)
  # #     d4 = copy.deepcopy(d)
  # #     d4[61] = 0.0
  # #     d4[62] = 0.0
  # #     d4[63] = 0.0
  # #     d4[64] = 1.0
  # #     x_test_final.append(d4)
  #

  model = snn(127,[64,32,16],1, optimizer='adam')
  model.load_model()
  T0, ST = model.survival_function(np.array(info),algo="wwe",base_X=np.array(x_train), base_label={'e':np.array(y_train['e']), 't':np.array(y_train['t'])})
  model.close()
  print([T0.tolist(),ST[0].tolist()])
  return [T0,ST]

# print(cur([1,1,1,0,0,1,0,0,1,0,0,0,0,0,1,0,1,0,0,0,0,0,1,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,1,1,0,0,0,1,0,0,0,0,1,1,0,0,0,1,0,0,0,0,1,0,0,1,0,0]))
  # model.load_data(np.array(x_train),{'e':np.array(y_train['e']), 't':np.array(y_train['t'])})
  # model.generate_graph()
  # model.train(np.array(x_train),{'e':np.array(y_train['e']), 't':np.array(y_train['t'])}, num_epoch=200, iteration=1)

  #
  # fp3 = open('results_LCCS.txt','w+')
  # fp3.write('id, Lobectomy, None, Sublobar, Peumonectomy')
  # fp3.write('\n')
  #
  # h0 = model.predict(np.array(x_test_final))
  # T0, ST = model.survival_function(np.array(x_test_final),algo="wwe")
  # for i in range(len(y_test_index)):
  #     fp3.write(str(y_test_index[i]))
  #     fp3.write(', ')
  #     fp3.write(str(-h0[i*4]))
  #     fp3.write(', ')
  #     fp3.write(str(-h0[i*4+1]))
  #     fp3.write(', ')
  #     fp3.write(str(-h0[i*4+2]))
  #     fp3.write(', ')
  #     fp3.write(str(-h0[i*4+3]))
  #     fp3.write('\n')
  #     vision.plot_surv_func(T0, ST[i*4:(i+1)*4], y_test_index[i])
  # fp3.close()
  # model.close()
# def json_loads_byteified(json_text):
#     return _byteify(
#         json.loads(json_text, object_hook=_byteify),
#         ignore_dicts=True
#     )
#
# def _byteify(data, ignore_dicts = False):
#     # if this is a unicode string, return its string representation
#     if isinstance(data, unicode):
#         return data.encode('utf-8')
#     # if this is a list of values, return list of byteified values
#     if isinstance(data, list):
#         return [ _byteify(item, ignore_dicts=True) for item in data ]
#     # if this is a dictionary, return dictionary of byteified keys and values
#     # but only if we haven't already byteified it
#     if isinstance(data, dict) and not ignore_dicts:
#         return {
#             _byteify(key, ignore_dicts=True): _byteify(value, ignore_dicts=True)
#             for key, value in data.iteritems()
#         }
#     # if it's anything else, return it in its original form
#     return data

if __name__ == '__main__':
    patient = sys.argv[1]
    # path = sys.argv[2]
    patient = json.loads(patient)
    # print(patient)
    cur([patient])
    # print(cur([1,1,1,0,0,1,0,0,1,0,0,0,0,0,1,0,1,0,0,0,0,0,1,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,1,1,0,0,0,1,0,0,0,0,1,1,0,0,0,1,0,0,0,0,1,0,0,1,0,0]))

