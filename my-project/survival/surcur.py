import tensorflow as tf
import numpy as np
import csv
from data_pre import *
import vision
from sklearn.preprocessing import MinMaxScaler
from lifelines.utils import concordance_index
from supersmoother import SuperSmoother

class snn(object):
    def __init__(self, X, label, input_node, hidden_layers_node, output_node,
                 learning_rate=0.001, learning_rate_decay=1.0, activation='tanh',
                 L2_reg=0.0, L1_reg=0.0, optimizer='sgd', dropout_keep_prob=1.0, seed=1):
        self.train_data = dict()
        self.train_data['X'], self.train_data['E'], \
            self.train_data['T'], self.train_data['failures'], \
            self.train_data['atrisk'], self.train_data['ties'], self.train_data['T_vector'], time_node = parse_data(X, label)
        print('pre success')
        G = tf.Graph()
        with G.as_default():
            tf.set_random_seed(seed)

            X = tf.placeholder(tf.float32, [None, input_node], name = 'x-input')
            y_ = tf.placeholder(tf.float32, [None, output_node], name = 'label-input')
            t_ = tf.placeholder(tf.float32, [None, time_node], name = 'time-input')
            keep_prob = tf.placeholder(tf.float32)

            self.nnweights = []
            self.nnbias = []
            prev_node = input_node
            prev_x = X
            for i in range(len(hidden_layers_node)):
                layer_name = 'layer' + str(i+1)
                with tf.variable_scope(layer_name, reuse=tf.AUTO_REUSE):
                    weights = tf.get_variable('weights',[prev_node,hidden_layers_node[i]],initializer=tf.truncated_normal_initializer(stddev=0.1))
                    self.nnweights.append(weights)

                    biases = tf.get_variable('biases',[hidden_layers_node[i]],initializer=tf.constant_initializer(0.0))
                    self.nnbias.append(biases)
                    layer_out = tf.nn.dropout(tf.matmul(prev_x,weights)+biases, keep_prob)

                    if activation == 'relu':
                        layer_out = tf.nn.relu(layer_out)
                    elif activation == 'sigmoid':
                        layer_out = tf.nn.sigmoid(layer_out)
                    elif activation == 'tanh':
                        layer_out = tf.nn.tanh(layer_out)
                    else:
                        raise NotImplementedError('activation not recognized')

                    prev_node = hidden_layers_node[i]
                    prev_x = layer_out

            layer_name = 'layer_last'
            with tf.variable_scope(layer_name, reuse=tf.AUTO_REUSE):
                weights = tf.get_variable('weights',[prev_node,output_node],initializer=tf.truncated_normal_initializer(stddev=0.1))
                self.nnweights.append(weights)
                biases = tf.get_variable('biases',[output_node],initializer=tf.constant_initializer(0.0))
                self.nnbias.append(biases)

                layer_out = tf.matmul(prev_x,weights)+biases
            y = layer_out
            layer_name = 'cur_last'
            with tf.variable_scope(layer_name, reuse=tf.AUTO_REUSE):
                k = tf.get_variable('k',[1],initializer=tf.constant_initializer(2.0))
                lamb = tf.get_variable('lamb',[1],initializer=tf.constant_initializer(1.0))
                time = []
                for p in range(time_node):
                    time.append((p+1)*1.0)
                time = tf.constant(time)
                time_out = (k/lamb)*tf.pow(time/lamb,k-1)*tf.exp(layer_out)
                time_out = tf.nn.sigmoid(time_out)

            y2 = time_out
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
            reg_item = tf.contrib.layers.l1_l2_regularizer(L1_reg, L2_reg)
            reg_term = tf.contrib.layers.apply_regularization(reg_item, self.nnweights)
            loss_fun = self._negative_log_likelihood(y_, y)
            loss_time = self._time_likelihood(y_, t_, y2)
            loss = loss_fun + reg_term + loss_time

            if optimizer == 'sgd':
                lr = tf.train.exponential_decay(learning_rate,global_step,1,learning_rate_decay)
                train_step = tf.train.GradientDescentOptimizer(lr).minimize(loss, global_step=global_step)
            elif optimizer == 'adam':
                train_step = tf.train.AdamOptimizer(learning_rate).minimize(loss, global_step=global_step)
            else:
                raise NotImplementedError('optimization not recognized')
            # init op
            init_op = tf.global_variables_initializer()

        print('graph success')
        self.X = X
        self.y_ = y_
        self.t_ = t_
        self.keep_prob = keep_prob
        self.y = y
        self.y2 = y2
        self.global_step = global_step
        self.loss = loss
        self.train_step = train_step
        self.configuration = {
            'input_node': input_node,
            'hidden_layers_node': hidden_layers_node,
            'output_node': output_node,
            'time_node': time_node,
            'learning_rate': learning_rate,
            'learning_rate_decay': learning_rate_decay,
            'activation': activation,
            'L1_reg': L1_reg,
            'L2_reg': L2_reg,
            'optimizer': optimizer,
            'dropout': dropout_keep_prob
        }

        self.sess = tf.Session(graph=G)
        self.sess.run(init_op)

    def train(self, num_epoch = 5000, iteration=-1):
        loss_list = []
        CI_list = []
        N = self.train_data['E'].shape[0]
        for i in range(num_epoch):
            _, output_y,output_time, loss_value, step = self.sess.run(
                    [self.train_step, self.y, self.y2, self.loss, self.global_step],
                    feed_dict= {
                        self.X: self.train_data['X'],
                        self.y_: self.train_data['E'].reshape((N,1)),
                        self.t_: self.train_data['T_vector'],
                        self.keep_prob: self.configuration['dropout']
                    }
                )
            print(output_time.shape)
            print(output_time[0])
            print(loss_value)
            loss_list.append(loss_value)
            label = {
                't': self.train_data['T'],
                'e': self.train_data['E']
            }
            CI = self._metrics_ci(label, output_y)
            CI_list.append(CI)
            if (iteration != -1) and (i % iteration == 0):
                print("-------------------------------------------------")
                print("training steps %d:\nloss = %g.\n" % (step, loss_value))
                print("CI = %g.\n" % CI)

    def _time_likelihood(self,y_true,t_vector,y_pred):
        logL = 0
        for i, e in enumerate(self.train_data['E']):
            if int(e) > 0:
                logL += tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=t_vector[i],logits=y_pred[i]))
            else:
                logL += tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=t_vector[i,:self.train_data['T'][i]],logits=y_pred[i,:self.train_data['T'][i]]))
        return logL


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
        print("session closed!")
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
        for i, v in enumerate(VIP):
            print("%dth feature score : %g." % (i, v))
        return VIP

    def survival_function(self, X, algo="wwe", base_X=None, base_label=None,
                          smoothed=False, is_plot=True):

        risk = self.predict(X)
        hazard_ratio = np.exp(risk.reshape((risk.shape[0], 1)))
        # Estimate S0(t) using data(base_X, base_label)
        T0, S0 = self.base_surv(algo=algo, X=base_X, label=base_label, smoothed=smoothed)
        ST = S0**(hazard_ratio)
        if is_plot:
            vision.plot_surv_func(T0, ST)
        return T0, ST

    def base_surv(self, algo="wwe", X=None, label=None, smoothed=False):

        if X is None or label is None:
            X = self.train_data['X']
            label = {'t': self.train_data['T'],
                     'e': self.train_data['E']}
        X, E, T, failures, atrisk, ties, t_vector, t_node = parse_data(X, label)

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

#load data
data = []
e = []
t = []
with open('./data/data.csv') as f:
    lines = csv.reader(f)
    for line in lines:
        data.append(line)

data = MinMaxScaler().fit_transform(data)

with open('./data/label.csv') as f:
    lines = csv.reader(f)
    for line in lines:
        e.append(float(line[1]))
        t.append(int(line[2]))
label = {'e': np.array(e[:1000]),'t': np.array(t[:1000])}
model = snn(np.array(data[:1000]),label,len(data[0]),[64,32,16],1)
model.train(num_epoch=100, iteration=1)
model.survival_function(np.array(data[:1]))
model.close()

