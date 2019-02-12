import sqlite3
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
# Create your connection.
cnx = sqlite3.connect('/Users/dns43/Downloads/compas.db')

df = pd.read_sql_query("SELECT * FROM compas", cnx)
df = df.drop_duplicates()
df.head()
df2 = pd.read_csv("https://github.com/propublica/compas-analysis/raw/master/compas-scores-two-years.csv", header=0).set_index('id')

corr = df2.corr()
sns.heatmap(corr, 
        xticklabels=corr.columns,
        yticklabels=corr.columns)

AA = df2[(df2['race'] == 'African-American')]
aa_total = AA.count()[1]
AA = AA[(AA['is_recid'] == 1.0)]
AA = AA[(AA['decile_score'] < 5)]
aa_fp = AA.count()[1]
print(aa_fp, aa_total)

C = df2[(df2['race'] == 'Caucasian')]
c_total = C.count()[1]
C = C[(C['is_recid'] == 1.0)]
C = C[(C['decile_score'] < 5)]
c_fp = C.count()[1]
print(c_fp, c_total)

print('False-Positive Rate for African-American: ', aa_fp/aa_total, 'vs Caucasian: ', c_fp/c_total)
print('=> gap of ', 1-((aa_fp/aa_total)/(c_fp/c_total)) )
df2 = df2.loc[df2['race'].isin(['African-American', 'Caucasian'])]
df2 = df2[['c_charge_degree', 'race', 'sex', 'is_recid', 'age', 'priors_count', 'juv_fel_count', 'juv_misd_count', 'juv_other_count']]
df2['age'] = df2['age']/100
df2['priors_count'] = df2['priors_count']/df2['priors_count'].max()
a,_= pd.factorize(df2['sex'])
df2['sex'] = a
a,_= pd.factorize(df2['c_charge_degree'])
df2['c_charge_degree'] = a
for i in df2:
    print(df2[i].value_counts(),'\n')


df2['is_recid'] = df2['is_recid'].replace(-1, 0)
df2.race.value_counts()
df2.is_recid.value_counts()

a,_= pd.factorize(df2['race'])
df2['race'] = a
print(df2.race.value_counts())
training_features = df2.sample(frac=0.8,random_state=0)
test_features = df2.drop(training_features.index)

training_label = pd.get_dummies(training_features['is_recid'])
training_features.pop('is_recid')

test_label = pd.get_dummies(test_features['is_recid'])
test_features.pop('is_recid')
dtest_label = pd.get_dummies(test_features['race'])
training_features.head(10)

training_features = training_features.values
#training_features = np.reshape(training_features, [len(training_features),2])
training_label = training_label.values
training_label = np.reshape(training_label, [len(training_label),2])
test_label = test_label.values
test_label = np.reshape(test_label, [len(test_label),2])

learning_rate = 0.0184
batch_size = 256
num_steps = int(len(training_label) / batch_size)
display_step = int(num_steps / 10)

# Network Parameters
n_hidden_1 = 256 # 1st layer number of neurons
n_hidden_2 = 256 # 2nd layer number of neurons
num_input = 8 # MNIST data input (img shape: 28*28)
num_classes = 2 # MNIST total classes (0-9 digits)

# tf Graph input
X = tf.placeholder("float", [None, num_input])
Y = tf.placeholder("float", [None, num_classes])

# Store layers weight & bias
weights = {
    'h1': tf.Variable(tf.random_normal([num_input, n_hidden_1])),
    'h2': tf.Variable(tf.random_normal([n_hidden_1, n_hidden_2])),
    'out': tf.Variable(tf.random_normal([n_hidden_2, num_classes]))
}
biases = {
    'b1': tf.Variable(tf.random_normal([n_hidden_1])),
    'b2': tf.Variable(tf.random_normal([n_hidden_2])),
    'out': tf.Variable(tf.random_normal([num_classes]))
}

# Create model
def neural_net(x):
    # Hidden fully connected layer with 256 neurons
    #layer_1 = tf.add(tf.matmul(x, weights['h1']), biases['b1'])
    layer_1 = tf.nn.relu(tf.add(tf.matmul(x, weights['h1']), biases['b1']))
    # Hidden fully connected layer with 256 neurons
    layer_2 = tf.nn.relu(tf.add(tf.matmul(layer_1, weights['h2']), biases['b2']))
    # Output fully connected layer with a neuron for each class
    out_layer = tf.matmul(layer_2, weights['out']) + biases['out']
    prob = tf.nn.sigmoid(out_layer)
    return out_layer, prob


# Construct model
logits, rprob = neural_net(X)

# Define loss and optimizer
loss_op = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=Y))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
train_op = optimizer.minimize(loss_op)

# Evaluate model (with test logits, for dropout to be disabled)
correct_pred = tf.equal(tf.argmax(logits, 1), tf.argmax(Y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
prediction = tf.argmax(logits, 1)

# Initialize the variables (i.e. assign their default value)
init = tf.global_variables_initializer()

# Start training
with tf.Session() as sess:

    # Run the initializer
    sess.run(init)
    #train = sess.
    #run(training_features)
    #train_labels = sess.run(training_labels)
    #test = sess.run(test_features)
    #test_labels = sess.run(test_labels)
    
    logit_output = []
    #prediction_output = []

    start = 0
    for step in range(1, num_steps+1):
        #batch_x, batch_y = training_data.next_batch(batch_size)
        # Run optimization op (backprop)
        start = start+batch_size
        end = start+batch_size
        
        sess.run(train_op, feed_dict={X: training_features[start:end], Y: training_label[start:end]})
        logit_output.append(sess.run(logits, feed_dict={X: training_features[start:end], Y: training_label[start:end]}))
        #prediction_output.append(sess.run(prediction, feed_dict={X: training_features[start:end], Y: training_label[start:end]}))
        if step % display_step == 0 or step == 1:
            # Calculate batch loss and accuracy
            loss, acc = sess.run([loss_op, accuracy], feed_dict={X: training_features[start:end],
                                                                 Y: training_label[start:end]})
            print("Step " + str(step) + ", Minibatch Loss= " + \
                  "{:.4f}".format(loss) + ", Training Accuracy= " + \
                  "{:.3f}".format(acc))

    print("Optimization Finished!")

    # Calculate accuracy for MNIST test images
    print("Testing Accuracy:", \
        sess.run(accuracy, feed_dict={X: test_features,
                                      Y: test_label}))
    prediction_output = sess.run(prediction, feed_dict={X: test_features, Y: test_label})
        

prediction_output
p = pd.DataFrame(prediction_output, columns=['p'])
p = pd.get_dummies(p['p'])

r = pd.DataFrame(test_label, columns=['is_recid0', 'is_recid1'])
d = pd.DataFrame(dtest_label.values, columns=['African-American', 'Caucasian'])
a = p.values
p = pd.DataFrame(a, columns=['p0', 'p1'])


dataset = pd.concat([r, d, p], axis=1, join_axes=[test_features.index])
dataset.head()

AA = dataset[['African-American', 'is_recid0','p1']]
C = dataset[['Caucasian', 'is_recid0','p1']]
#'African-American', 'is_recid0','p'
#df[ (df['A']>0) & (df['B']>0) & (df['C']>0)]
#AA.all(1.0)
#AA[ ((AA['African-American'] == 1.0) ^ (AA['is_recid0'] == 1.0) ^ (AA['p'] == 1.0)).all() ]
AA = AA[(AA['African-American'] == 1.0)]
aa_total = AA.count()[1]
AA = AA[(AA['is_recid0'] == 1.0)]
AA = AA[(AA['p1'] == 1.0)]
aa_fp = AA.count()[1]
print(aa_fp)

C = C[(C['Caucasian'] == 1.0)]
c_total = C.count()[1]
C = C[(C['is_recid0'] == 1.0)]
C = C[(C['p1'] == 1.0)]
c_fp = C.count()[1]
print(c_fp)

print('False-Positive Rate for African-American: ', aa_fp/aa_total, 'vs Caucasian: ', c_fp/c_total)
print('=> gap of ', 1-((aa_fp/aa_total)/(c_fp/c_total)) )

