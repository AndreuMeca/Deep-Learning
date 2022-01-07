# Assignment 3: Attention

The objectives of this assignment are:

+ To implement Bahdanau Attention and Luong General Attention classes.
+ To do a comparative (# of steps to converge, test error) of the three methods we have seen. Use these values for the comparative (the training datset size and `rnn_units` and `batch_size` values are up to you): 
    + `n_timesteps_in = 100`
    + `n_features = 20`.   
+ To implement a function to visualize the attention weights for one example. You can visualize them as in this figure (that corresponds to a machine translation task):

<div>
<center>
<img src="https://jalammar.github.io/images/attention_sentence.png" width="200">
</center>
</div>

+ To write a blog entry explaining in your words how does attention work. You can do it in your favourite blog site. If you do not have a favourite blog site, you can start one here: https://hackmd.io/

You have to report all your work at the end of this notebook.



# Code


```python
import random
random.seed(123)
```


```python
#@title Some utils
from random import randint
from matplotlib import pyplot as plt
from tensorflow.keras.callbacks import EarlyStopping

import numpy as np

def generate_sequence(length, n_unique):
    """
    Generate a sequence of random integers.
    
    :length: Total length of the generated sequence
    :n_unique: Maximum number allowed
    """
    return [randint(1, n_unique-1) for _ in range(length)]

def one_hot_encode(sequence, n_unique):
    """
    Transform a sequence of integers into a one-hot-encoding vector
    
    :sequence: The sequence we want to transform
    :n_unique: Maximum number allowed (length of the one-hot-encoded vector)
    """
    encoding = list()
    for value in sequence:
        vector = [0 for _ in range(n_unique)]
        vector[value] = 1
        encoding.append(vector)
    return np.array(encoding)

def one_hot_decode(encoded_seq):
    """
    Transorm a one-hot-encoded vector into a list of integers
    
    :encoded_seq: One hot encoded sequence to be transformed
    """
    return [np.argmax(vector) for vector in encoded_seq]


def get_reversed_pairs(time_steps,vocabulary_size):
    """
    Generate a pair X, y where y is the 'reversed' version of X.
    
    :time_steps: Sequence length
    :vocabulary_size: Maximum number allowed
    """
    # generate random sequence and reverse it
    sequence_in = generate_sequence(time_steps, vocabulary_size)
    sequence_out = sequence_in[::-1]

    # one hot encode both sequences
    X = one_hot_encode(sequence_in, vocabulary_size)
    y = one_hot_encode(sequence_out, vocabulary_size)
    
    # reshape as 3D so it can be inputed to the LSTM
    X = X.reshape((1, X.shape[0], X.shape[1]))
    y = y.reshape((1, y.shape[0], y.shape[1]))
    return X,y


def create_dataset(train_size, test_size, time_steps,vocabulary_size):
    """
    Generates a datset of reversed pairs X, y.
    
    :train_size: Number of train pairs
    :test_size: Number of test pairs
    :time_steps: Sequence length
    :vocabulary_size: Maximum number allowed
    """
    
    # Generate reversed pairs for training
    pairs = [get_reversed_pairs(time_steps,vocabulary_size) for _ in range(train_size)]
    pairs= np.array(pairs).squeeze()
    X_train = pairs[:,0]
    y_train = pairs[:,1]
    
    # Generate reversed pairs for test
    pairs = [get_reversed_pairs(time_steps,vocabulary_size) for _ in range(test_size)]
    pairs= np.array(pairs).squeeze()
    X_test = pairs[:,0]
    y_test = pairs[:,1]	

    return X_train, y_train, X_test, y_test


def train_test(model, X_train, y_train , X_test, y_test, epochs=500, batch_size=32, patience=5):
    """
    It trains a model and evaluates the result on the test dataset
    
    :model: Model to be fit
    :X_train, y_train: Train samples and labels 
    :X_test y_test: Test samples and labels 
    :epochs: Maximum number of iterations that the model will perform
    :batch_size: Samples per batch
    :patience: Number of rounds without improvement that the model can perform. If there is no improvement on the loss, it will stop the trainning process.
    """
    
    # Train the model
    history=model.fit(X_train, y_train, 
                      validation_split= 0.1, 
                      epochs=epochs,
                      batch_size=batch_size, 
                      callbacks=[EarlyStopping(monitor='val_loss', patience=patience, restore_best_weights=True)])
    
    _, train_acc = model.evaluate(X_train, y_train, batch_size=batch_size)
    _, test_acc = model.evaluate(X_test, y_test, batch_size=batch_size)
    
    print('\nPREDICTION ACCURACY (%):')
    print('Train: %.3f, Test: %.3f' % (train_acc*100, test_acc*100))
    
    fig, axs = plt.subplots(1,2, figsize=(12,5))
    # summarize history for loss
    axs[0].plot(history.history['loss'])
    axs[0].plot(history.history['val_loss'])
    axs[0].set_title(model.name+' loss')
    axs[0].set_ylabel('loss')
    axs[0].set_xlabel('epoch')
    axs[0].legend(['train', 'val'], loc='upper left')
    
    # summarize history for accuracy
    axs[1].plot(history.history['accuracy'])
    axs[1].plot(history.history['val_accuracy'])
    axs[1].set_title(model.name+' accuracy')
    axs[1].set_ylabel('accuracy')
    axs[1].set_xlabel('epoch')
    axs[1].legend(['train', 'val'], loc='upper left')
    plt.show()
    
    
def predict(model, n_timesteps_in,n_features, x, y_real=None, ):
    pred=model.predict(x.reshape(1,n_timesteps_in,n_features), batch_size=1)
    print('input', one_hot_decode(x))    
    print('predicted', one_hot_decode(pred[0]))
    if y_real is not None:
        print('expected', one_hot_decode(y_real))
```


```python
import tensorflow as tf
import numpy as np

from tensorflow.keras.layers import Input, Dense, LSTM, Lambda
from tensorflow.keras.models import Model
from tensorflow.keras import backend as K

tf.keras.backend.set_floatx('float64')

#attention model
def build_attention_model(attention, batch_size, rnn_units):
    
    # ENCODER STEP
    # ------------
    # Same encoder as before with one and only difference. Now we need all the lstm states so we
    # set return_sequences=True and return_state=True.
    encoder_inputs = Input(shape=(n_timesteps_in, n_features), name='encoder_inputs')
    encoder_lstm = LSTM(rnn_units, return_sequences=True, return_state=True, name='encoder_lstm')
    encoder_outputs, encoder_state_h, encoder_state_c = encoder_lstm(encoder_inputs)
    
    states = [encoder_state_h, encoder_state_c]
    
    # DECODER STEP
    # ------------
    # Set up the decoder layers
    # input shape: (1, n_features + rnn_units)
    decoder_lstm = LSTM(rnn_units, return_state=True, name='decoder_lstm')
    decoder_dense = Dense(n_features, activation='softmax', name='decoder_dense')
    
    # As before, we use as first input the 0-sequence
    all_outputs = []
    inputs = np.zeros((batch_size, 1, n_features))
    
    # Decoder_outputs is the last hidden state of the encoder. Encoder_outputs are all the states
    decoder_outputs = encoder_state_h
    
    # Decoder will only process one time step at a time.
    for _ in range(n_timesteps_in):

        # Pay attention!
        # decoder_outputs (last hidden state) + encoder_outputs (all hidden states)
        context_vector, attention_weights = attention(decoder_outputs, encoder_outputs)
        context_vector = tf.expand_dims(context_vector, 1)

        # create the context vector by applying attention to 
        # Concatenate the input + context vectore to find the next decoder's input
        inputs = tf.concat([context_vector, inputs], axis=-1)

        # Passing the concatenated vector to the LSTM
        # Run the decoder on one timestep with attended input and previous states
        decoder_outputs, state_h, state_c = decoder_lstm(inputs, initial_state=states)        
        outputs = decoder_dense(decoder_outputs)
        
        # Use the last hidden state for prediction the output
        # save the current prediction
        # we will concatenate all predictions later
        outputs = tf.expand_dims(outputs, 1)
        all_outputs.append(outputs)
        
        # Reinject the output (prediction) as inputs for the next loop iteration
        # as well as update the states
        inputs = outputs
        states = [state_h, state_c]
        
    decoder_outputs = Lambda(lambda x: K.concatenate(x, axis=1))(all_outputs)
    model = Model(encoder_inputs, decoder_outputs, name='model_encoder_decoder')
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    
    return model
```


```python
# dataset 
n_timesteps_in = 100  # Sequence length
n_features = 20     # Maximum number allowed-1 (length of the one-hot-encoded vector)
train_size = 2000 
test_size = 200
X_train, y_train, X_test, y_test = create_dataset(train_size, test_size, n_timesteps_in,n_features)

# training parameters
batch_size = 100

# model parameters
rnn_units = 100
```


```python
class LuongDotAttention(tf.keras.layers.Layer):
    def __init__(self):
        super(LuongDotAttention, self).__init__()

    def call(self, query, values):
        query_with_time_axis = tf.expand_dims(query, 1)
        values_transposed = tf.transpose(values, perm=[0, 2, 1])

        # LUONGH Dot-product
        score = tf.transpose(tf.matmul(query_with_time_axis, 
                                       values_transposed), perm=[0, 2, 1])

        # attention_weights shape == (batch_size, max_length, 1)
        attention_weights = tf.nn.softmax(score, axis=1)
        
        # context_vector shape after sum == (batch_size, hidden_size)
        context_vector = attention_weights * values
        context_vector = tf.reduce_sum(context_vector, axis=1)

        return context_vector, attention_weights
```


```python
# attention model
attention = LuongDotAttention()
model_attention = build_attention_model(attention, batch_size, rnn_units)
```


```python
#training
train_test(model_attention, X_train, y_train , X_test,
           y_test, batch_size=batch_size, epochs=50, patience=3)
```

    Epoch 1/50
    18/18 [==============================] - 209s 3s/step - loss: 2.9633 - accuracy: 0.0542 - val_loss: 2.9467 - val_accuracy: 0.0522
    Epoch 2/50
    18/18 [==============================] - 12s 681ms/step - loss: 2.9445 - accuracy: 0.0585 - val_loss: 2.9418 - val_accuracy: 0.0631
    Epoch 3/50
    18/18 [==============================] - 12s 694ms/step - loss: 2.9391 - accuracy: 0.0629 - val_loss: 2.9368 - val_accuracy: 0.0640
    Epoch 4/50
    18/18 [==============================] - 12s 694ms/step - loss: 2.9319 - accuracy: 0.0708 - val_loss: 2.9272 - val_accuracy: 0.0711
    Epoch 5/50
    18/18 [==============================] - 12s 680ms/step - loss: 2.9251 - accuracy: 0.0766 - val_loss: 2.9202 - val_accuracy: 0.0790
    Epoch 6/50
    18/18 [==============================] - 12s 682ms/step - loss: 2.9173 - accuracy: 0.0813 - val_loss: 2.9124 - val_accuracy: 0.0841
    Epoch 7/50
    18/18 [==============================] - 12s 679ms/step - loss: 2.9122 - accuracy: 0.0841 - val_loss: 2.9054 - val_accuracy: 0.0854
    Epoch 8/50
    18/18 [==============================] - 12s 680ms/step - loss: 2.9048 - accuracy: 0.0880 - val_loss: 2.8977 - val_accuracy: 0.0908
    Epoch 9/50
    18/18 [==============================] - 12s 693ms/step - loss: 2.8961 - accuracy: 0.0918 - val_loss: 2.8919 - val_accuracy: 0.0931
    Epoch 10/50
    18/18 [==============================] - 12s 694ms/step - loss: 2.8847 - accuracy: 0.0958 - val_loss: 2.8786 - val_accuracy: 0.0964
    Epoch 11/50
    18/18 [==============================] - 12s 693ms/step - loss: 2.8688 - accuracy: 0.1027 - val_loss: 2.8624 - val_accuracy: 0.1059
    Epoch 12/50
    18/18 [==============================] - 12s 692ms/step - loss: 2.8480 - accuracy: 0.1096 - val_loss: 2.8360 - val_accuracy: 0.1113
    Epoch 13/50
    18/18 [==============================] - 12s 680ms/step - loss: 2.8147 - accuracy: 0.1222 - val_loss: 2.8012 - val_accuracy: 0.1250
    Epoch 14/50
    18/18 [==============================] - 12s 679ms/step - loss: 2.7704 - accuracy: 0.1357 - val_loss: 2.7468 - val_accuracy: 0.1441
    Epoch 15/50
    18/18 [==============================] - 12s 681ms/step - loss: 2.7218 - accuracy: 0.1507 - val_loss: 2.7000 - val_accuracy: 0.1563
    Epoch 16/50
    18/18 [==============================] - 12s 683ms/step - loss: 2.6807 - accuracy: 0.1615 - val_loss: 2.6801 - val_accuracy: 0.1613
    Epoch 17/50
    18/18 [==============================] - 12s 679ms/step - loss: 2.6362 - accuracy: 0.1725 - val_loss: 2.6126 - val_accuracy: 0.1799
    Epoch 18/50
    18/18 [==============================] - 12s 693ms/step - loss: 2.5616 - accuracy: 0.1897 - val_loss: 2.5137 - val_accuracy: 0.2025
    Epoch 19/50
    18/18 [==============================] - 12s 681ms/step - loss: 2.5164 - accuracy: 0.1998 - val_loss: 2.4959 - val_accuracy: 0.2056
    Epoch 20/50
    18/18 [==============================] - 12s 680ms/step - loss: 2.4689 - accuracy: 0.2114 - val_loss: 2.4580 - val_accuracy: 0.2099
    Epoch 21/50
    18/18 [==============================] - 12s 679ms/step - loss: 2.3955 - accuracy: 0.2278 - val_loss: 2.4349 - val_accuracy: 0.2114
    Epoch 22/50
    18/18 [==============================] - 12s 679ms/step - loss: 2.3838 - accuracy: 0.2274 - val_loss: 2.3099 - val_accuracy: 0.2439
    Epoch 23/50
    18/18 [==============================] - 12s 679ms/step - loss: 2.4617 - accuracy: 0.2102 - val_loss: 2.3360 - val_accuracy: 0.2364
    Epoch 24/50
    18/18 [==============================] - 12s 679ms/step - loss: 2.3028 - accuracy: 0.2465 - val_loss: 2.2708 - val_accuracy: 0.2480
    Epoch 25/50
    18/18 [==============================] - 12s 679ms/step - loss: 2.2216 - accuracy: 0.2609 - val_loss: 2.1918 - val_accuracy: 0.2661
    Epoch 26/50
    18/18 [==============================] - 12s 695ms/step - loss: 2.2797 - accuracy: 0.2457 - val_loss: 2.2186 - val_accuracy: 0.2626
    Epoch 27/50
    18/18 [==============================] - 12s 680ms/step - loss: 2.1589 - accuracy: 0.2730 - val_loss: 2.1150 - val_accuracy: 0.2798
    Epoch 28/50
    18/18 [==============================] - 12s 681ms/step - loss: 2.1111 - accuracy: 0.2823 - val_loss: 2.0645 - val_accuracy: 0.2903
    Epoch 29/50
    18/18 [==============================] - 12s 695ms/step - loss: 2.0710 - accuracy: 0.2902 - val_loss: 2.0487 - val_accuracy: 0.2887
    Epoch 30/50
    18/18 [==============================] - 12s 694ms/step - loss: 2.0116 - accuracy: 0.3024 - val_loss: 1.9732 - val_accuracy: 0.3071
    Epoch 31/50
    18/18 [==============================] - 12s 695ms/step - loss: 1.9404 - accuracy: 0.3185 - val_loss: 1.9478 - val_accuracy: 0.3140
    Epoch 32/50
    18/18 [==============================] - 12s 695ms/step - loss: 1.9271 - accuracy: 0.3186 - val_loss: 1.8895 - val_accuracy: 0.3263
    Epoch 33/50
    18/18 [==============================] - 12s 680ms/step - loss: 1.9729 - accuracy: 0.3045 - val_loss: 1.8974 - val_accuracy: 0.3195
    Epoch 34/50
    18/18 [==============================] - 12s 694ms/step - loss: 1.8364 - accuracy: 0.3387 - val_loss: 1.7758 - val_accuracy: 0.3597
    Epoch 35/50
    18/18 [==============================] - 12s 681ms/step - loss: 1.7605 - accuracy: 0.3570 - val_loss: 1.7307 - val_accuracy: 0.3688
    Epoch 36/50
    18/18 [==============================] - 12s 679ms/step - loss: 1.7531 - accuracy: 0.3571 - val_loss: 1.7938 - val_accuracy: 0.3394
    Epoch 37/50
    18/18 [==============================] - 12s 677ms/step - loss: 1.6930 - accuracy: 0.3745 - val_loss: 1.6647 - val_accuracy: 0.3836
    Epoch 38/50
    18/18 [==============================] - 12s 677ms/step - loss: 1.6849 - accuracy: 0.3717 - val_loss: 1.6844 - val_accuracy: 0.3629
    Epoch 39/50
    18/18 [==============================] - 12s 679ms/step - loss: 1.6110 - accuracy: 0.3963 - val_loss: 1.6249 - val_accuracy: 0.3855
    Epoch 40/50
    18/18 [==============================] - 12s 679ms/step - loss: 1.5736 - accuracy: 0.4076 - val_loss: 1.5165 - val_accuracy: 0.4283
    Epoch 41/50
    18/18 [==============================] - 12s 678ms/step - loss: 1.6070 - accuracy: 0.3906 - val_loss: 1.7096 - val_accuracy: 0.3538
    Epoch 42/50
    18/18 [==============================] - 12s 678ms/step - loss: 1.5381 - accuracy: 0.4112 - val_loss: 1.5226 - val_accuracy: 0.4155
    Epoch 43/50
    18/18 [==============================] - 12s 681ms/step - loss: 1.4754 - accuracy: 0.4337 - val_loss: 1.4163 - val_accuracy: 0.4573
    Epoch 44/50
    18/18 [==============================] - 12s 691ms/step - loss: 1.4700 - accuracy: 0.4337 - val_loss: 1.4679 - val_accuracy: 0.4411
    Epoch 45/50
    18/18 [==============================] - 12s 681ms/step - loss: 1.3972 - accuracy: 0.4609 - val_loss: 1.2995 - val_accuracy: 0.5046
    Epoch 46/50
    18/18 [==============================] - 12s 682ms/step - loss: 1.3621 - accuracy: 0.4756 - val_loss: 1.2508 - val_accuracy: 0.5264
    Epoch 47/50
    18/18 [==============================] - 12s 695ms/step - loss: 1.3213 - accuracy: 0.4877 - val_loss: 1.3465 - val_accuracy: 0.4729
    Epoch 48/50
    18/18 [==============================] - 12s 695ms/step - loss: 1.2757 - accuracy: 0.5092 - val_loss: 1.6954 - val_accuracy: 0.3367
    Epoch 49/50
    18/18 [==============================] - 12s 681ms/step - loss: 1.3460 - accuracy: 0.4778 - val_loss: 1.2747 - val_accuracy: 0.4984
    20/20 [==============================] - 4s 205ms/step - loss: 1.2391 - accuracy: 0.5317
    2/2 [==============================] - 0s 206ms/step - loss: 1.2524 - accuracy: 0.5229
    
    PREDICTION ACCURACY (%):
    Train: 53.171, Test: 52.290



    
![png](output_8_1.png)
    



```python
class BahdanauAttention(tf.keras.layers.Layer):
    
    def __init__(self, units):
        super(BahdanauAttention, self).__init__()

        ##################
        # YOUR CODE HERE #
        self.W1 = tf.keras.layers.Dense(units)
        self.W2 = tf.keras.layers.Dense(units)
        self.V = tf.keras.layers.Dense(1)
        ##################

    def call(self, query, values):

        ##################
        # YOUR CODE HERE #
        query_with_time_axis = tf.expand_dims(query, 1)
        score = self.V(tf.nn.tanh(self.W1(query_with_time_axis) + self.W2(values)))
        attention_weights = tf.nn.softmax(score, axis=1)
        context_vector = attention_weights * values
        context_vector = tf.reduce_sum(context_vector, axis=1)
        ##################

        return context_vector, attention_weights
```


```python
# attention model
attention = BahdanauAttention(rnn_units)
model_attention_bahdanau = build_attention_model(attention, batch_size, rnn_units)
```


```python
#training
train_test(model_attention_bahdanau, X_train, y_train , X_test,
           y_test, batch_size=batch_size, epochs=50, patience=3)
```

    Epoch 1/50
    18/18 [==============================] - 218s 4s/step - loss: 2.9639 - accuracy: 0.0535 - val_loss: 2.9460 - val_accuracy: 0.0578
    Epoch 2/50
    18/18 [==============================] - 14s 780ms/step - loss: 2.9452 - accuracy: 0.0545 - val_loss: 2.9434 - val_accuracy: 0.0573
    Epoch 3/50
    18/18 [==============================] - 14s 777ms/step - loss: 2.9423 - accuracy: 0.0596 - val_loss: 2.9386 - val_accuracy: 0.0598
    Epoch 4/50
    18/18 [==============================] - 14s 782ms/step - loss: 2.9334 - accuracy: 0.0693 - val_loss: 2.9264 - val_accuracy: 0.0757
    Epoch 5/50
    18/18 [==============================] - 14s 771ms/step - loss: 2.9237 - accuracy: 0.0761 - val_loss: 2.9192 - val_accuracy: 0.0819
    Epoch 6/50
    18/18 [==============================] - 14s 782ms/step - loss: 2.9152 - accuracy: 0.0808 - val_loss: 2.9122 - val_accuracy: 0.0856
    Epoch 7/50
    18/18 [==============================] - 14s 783ms/step - loss: 2.9077 - accuracy: 0.0857 - val_loss: 2.9048 - val_accuracy: 0.0883
    Epoch 8/50
    18/18 [==============================] - 14s 775ms/step - loss: 2.9018 - accuracy: 0.0882 - val_loss: 2.8989 - val_accuracy: 0.0910
    Epoch 9/50
    18/18 [==============================] - 14s 783ms/step - loss: 2.8974 - accuracy: 0.0895 - val_loss: 2.9020 - val_accuracy: 0.0882
    Epoch 10/50
    18/18 [==============================] - 14s 773ms/step - loss: 2.8939 - accuracy: 0.0913 - val_loss: 2.8872 - val_accuracy: 0.0920
    Epoch 11/50
    18/18 [==============================] - 14s 776ms/step - loss: 2.8812 - accuracy: 0.0968 - val_loss: 2.8765 - val_accuracy: 0.0965
    Epoch 12/50
    18/18 [==============================] - 14s 785ms/step - loss: 2.8690 - accuracy: 0.1010 - val_loss: 2.8622 - val_accuracy: 0.1014
    Epoch 13/50
    18/18 [==============================] - 14s 784ms/step - loss: 2.8536 - accuracy: 0.1060 - val_loss: 2.8449 - val_accuracy: 0.1106
    Epoch 14/50
    18/18 [==============================] - 14s 775ms/step - loss: 2.8296 - accuracy: 0.1148 - val_loss: 2.8060 - val_accuracy: 0.1207
    Epoch 15/50
    18/18 [==============================] - 14s 776ms/step - loss: 2.7919 - accuracy: 0.1282 - val_loss: 2.7642 - val_accuracy: 0.1392
    Epoch 16/50
    18/18 [==============================] - 14s 782ms/step - loss: 2.7321 - accuracy: 0.1473 - val_loss: 2.6866 - val_accuracy: 0.1579
    Epoch 17/50
    18/18 [==============================] - 14s 784ms/step - loss: 2.6608 - accuracy: 0.1662 - val_loss: 2.6355 - val_accuracy: 0.1689
    Epoch 18/50
    18/18 [==============================] - 14s 783ms/step - loss: 2.6150 - accuracy: 0.1766 - val_loss: 2.5712 - val_accuracy: 0.1875
    Epoch 19/50
    18/18 [==============================] - 14s 776ms/step - loss: 2.5565 - accuracy: 0.1894 - val_loss: 2.5279 - val_accuracy: 0.1979
    Epoch 20/50
    18/18 [==============================] - 14s 775ms/step - loss: 2.4777 - accuracy: 0.2080 - val_loss: 2.5065 - val_accuracy: 0.1986
    Epoch 21/50
    18/18 [==============================] - 14s 785ms/step - loss: 2.4214 - accuracy: 0.2198 - val_loss: 2.3567 - val_accuracy: 0.2404
    Epoch 22/50
    18/18 [==============================] - 14s 773ms/step - loss: 2.4547 - accuracy: 0.2105 - val_loss: 2.4135 - val_accuracy: 0.2205
    Epoch 23/50
    18/18 [==============================] - 14s 781ms/step - loss: 2.3277 - accuracy: 0.2430 - val_loss: 2.4364 - val_accuracy: 0.2087
    Epoch 24/50
    18/18 [==============================] - 14s 781ms/step - loss: 2.4142 - accuracy: 0.2179 - val_loss: 2.4149 - val_accuracy: 0.2193
    20/20 [==============================] - 6s 272ms/step - loss: 2.3509 - accuracy: 0.2380
    2/2 [==============================] - 1s 321ms/step - loss: 2.3607 - accuracy: 0.2344
    
    PREDICTION ACCURACY (%):
    Train: 23.803, Test: 23.440



    
![png](output_11_1.png)
    



```python
class LuongGeneralAttention(tf.keras.layers.Layer):
    def __init__(self, units):
        super(LuongGeneralAttention, self).__init__()
        
        ##################
        # YOUR CODE HERE #
        self.W = tf.keras.layers.Dense(units)
        ##################

    def call(self, query, values):
        query_with_time_axis = tf.expand_dims(query, 1)
        values_transposed = tf.transpose(values, perm=[0, 2, 1])

        ##################
        # YOUR CODE HERE #
        score = tf.transpose(tf.matmul((query_with_time_axis), 
                                       self.W(values_transposed)), perm=[0, 2, 1])

        attention_weights = tf.nn.softmax(score, axis=1)
        
        context_vector = attention_weights * values
        context_vector = tf.reduce_sum(context_vector, axis=1)
        ##################

        return context_vector, attention_weights
```


```python
# attention model
attention = LuongGeneralAttention(rnn_units)
model_attention = build_attention_model(attention, batch_size, rnn_units)
```


```python
#training
train_test(model_attention, X_train, y_train , X_test,
           y_test, batch_size=batch_size, epochs=50, patience=3)
```

    Epoch 1/50
    18/18 [==============================] - 212s 3s/step - loss: 2.9618 - accuracy: 0.0529 - val_loss: 2.9456 - val_accuracy: 0.0537
    Epoch 2/50
    18/18 [==============================] - 13s 710ms/step - loss: 2.9441 - accuracy: 0.0583 - val_loss: 2.9437 - val_accuracy: 0.0539
    Epoch 3/50
    18/18 [==============================] - 12s 682ms/step - loss: 2.9400 - accuracy: 0.0642 - val_loss: 2.9357 - val_accuracy: 0.0692
    Epoch 4/50
    18/18 [==============================] - 13s 697ms/step - loss: 2.9347 - accuracy: 0.0684 - val_loss: 2.9303 - val_accuracy: 0.0712
    Epoch 5/50
    18/18 [==============================] - 12s 683ms/step - loss: 2.9204 - accuracy: 0.0794 - val_loss: 2.9090 - val_accuracy: 0.0868
    Epoch 6/50
    18/18 [==============================] - 12s 681ms/step - loss: 2.8992 - accuracy: 0.0920 - val_loss: 2.9075 - val_accuracy: 0.0857
    Epoch 7/50
    18/18 [==============================] - 12s 696ms/step - loss: 2.8792 - accuracy: 0.0994 - val_loss: 2.8568 - val_accuracy: 0.1083
    Epoch 8/50
    18/18 [==============================] - 12s 681ms/step - loss: 2.8516 - accuracy: 0.1097 - val_loss: 2.8366 - val_accuracy: 0.1184
    Epoch 9/50
    18/18 [==============================] - 12s 682ms/step - loss: 2.8126 - accuracy: 0.1221 - val_loss: 2.8590 - val_accuracy: 0.1091
    Epoch 10/50
    18/18 [==============================] - 12s 682ms/step - loss: 2.8242 - accuracy: 0.1186 - val_loss: 2.7804 - val_accuracy: 0.1285
    Epoch 11/50
    18/18 [==============================] - 12s 695ms/step - loss: 2.7607 - accuracy: 0.1366 - val_loss: 2.7450 - val_accuracy: 0.1426
    Epoch 12/50
    18/18 [==============================] - 12s 696ms/step - loss: 2.7205 - accuracy: 0.1484 - val_loss: 2.6883 - val_accuracy: 0.1582
    Epoch 13/50
    18/18 [==============================] - 12s 683ms/step - loss: 2.6897 - accuracy: 0.1554 - val_loss: 2.6622 - val_accuracy: 0.1618
    Epoch 14/50
    18/18 [==============================] - 12s 695ms/step - loss: 2.6281 - accuracy: 0.1705 - val_loss: 2.7270 - val_accuracy: 0.1431
    Epoch 15/50
    18/18 [==============================] - 12s 697ms/step - loss: 2.6144 - accuracy: 0.1729 - val_loss: 2.5585 - val_accuracy: 0.1863
    Epoch 16/50
    18/18 [==============================] - 12s 682ms/step - loss: 2.5554 - accuracy: 0.1857 - val_loss: 2.6664 - val_accuracy: 0.1587
    Epoch 17/50
    18/18 [==============================] - 12s 682ms/step - loss: 2.5643 - accuracy: 0.1853 - val_loss: 2.5069 - val_accuracy: 0.1976
    Epoch 18/50
    18/18 [==============================] - 12s 682ms/step - loss: 2.4630 - accuracy: 0.2071 - val_loss: 2.4346 - val_accuracy: 0.2132
    Epoch 19/50
    18/18 [==============================] - 12s 683ms/step - loss: 2.5574 - accuracy: 0.1847 - val_loss: 2.4795 - val_accuracy: 0.2057
    Epoch 20/50
    18/18 [==============================] - 12s 696ms/step - loss: 2.4398 - accuracy: 0.2117 - val_loss: 2.4121 - val_accuracy: 0.2155
    Epoch 21/50
    18/18 [==============================] - 12s 684ms/step - loss: 2.3676 - accuracy: 0.2268 - val_loss: 2.3434 - val_accuracy: 0.2293
    Epoch 22/50
    18/18 [==============================] - 12s 696ms/step - loss: 2.4042 - accuracy: 0.2162 - val_loss: 2.3624 - val_accuracy: 0.2284
    Epoch 23/50
    18/18 [==============================] - 12s 682ms/step - loss: 2.3175 - accuracy: 0.2360 - val_loss: 2.3056 - val_accuracy: 0.2334
    Epoch 24/50
    18/18 [==============================] - 12s 696ms/step - loss: 2.2737 - accuracy: 0.2419 - val_loss: 2.2453 - val_accuracy: 0.2460
    Epoch 25/50
    18/18 [==============================] - 12s 696ms/step - loss: 2.2532 - accuracy: 0.2446 - val_loss: 2.3542 - val_accuracy: 0.2140
    Epoch 26/50
    18/18 [==============================] - 12s 684ms/step - loss: 2.2673 - accuracy: 0.2383 - val_loss: 2.2010 - val_accuracy: 0.2510
    Epoch 27/50
    18/18 [==============================] - 12s 696ms/step - loss: 2.1980 - accuracy: 0.2520 - val_loss: 2.2957 - val_accuracy: 0.2251
    Epoch 28/50
    18/18 [==============================] - 12s 681ms/step - loss: 2.2185 - accuracy: 0.2481 - val_loss: 2.2226 - val_accuracy: 0.2465
    Epoch 29/50
    18/18 [==============================] - 12s 682ms/step - loss: 2.1358 - accuracy: 0.2654 - val_loss: 2.1050 - val_accuracy: 0.2708
    Epoch 30/50
    18/18 [==============================] - 12s 682ms/step - loss: 2.0820 - accuracy: 0.2773 - val_loss: 2.0679 - val_accuracy: 0.2768
    Epoch 31/50
    18/18 [==============================] - 12s 683ms/step - loss: 2.0667 - accuracy: 0.2781 - val_loss: 2.1242 - val_accuracy: 0.2606
    Epoch 32/50
    18/18 [==============================] - 12s 696ms/step - loss: 2.0729 - accuracy: 0.2741 - val_loss: 2.0317 - val_accuracy: 0.2821
    Epoch 33/50
    18/18 [==============================] - 12s 683ms/step - loss: 2.0243 - accuracy: 0.2848 - val_loss: 1.9933 - val_accuracy: 0.2898
    Epoch 34/50
    18/18 [==============================] - 12s 682ms/step - loss: 1.9825 - accuracy: 0.2933 - val_loss: 1.9688 - val_accuracy: 0.2940
    Epoch 35/50
    18/18 [==============================] - 12s 696ms/step - loss: 1.9726 - accuracy: 0.2935 - val_loss: 1.9341 - val_accuracy: 0.2979
    Epoch 36/50
    18/18 [==============================] - 12s 682ms/step - loss: 1.9220 - accuracy: 0.3046 - val_loss: 1.9033 - val_accuracy: 0.3056
    Epoch 37/50
    18/18 [==============================] - 12s 683ms/step - loss: 1.9274 - accuracy: 0.3017 - val_loss: 1.9344 - val_accuracy: 0.2984
    Epoch 38/50
    18/18 [==============================] - 12s 682ms/step - loss: 1.9014 - accuracy: 0.3088 - val_loss: 1.8681 - val_accuracy: 0.3137
    Epoch 39/50
    18/18 [==============================] - 12s 683ms/step - loss: 1.8371 - accuracy: 0.3235 - val_loss: 1.8367 - val_accuracy: 0.3200
    Epoch 40/50
    18/18 [==============================] - 12s 683ms/step - loss: 1.8420 - accuracy: 0.3199 - val_loss: 1.8958 - val_accuracy: 0.3048
    Epoch 41/50
    18/18 [==============================] - 12s 682ms/step - loss: 1.8199 - accuracy: 0.3233 - val_loss: 1.8191 - val_accuracy: 0.3208
    Epoch 42/50
    18/18 [==============================] - 13s 697ms/step - loss: 1.7388 - accuracy: 0.3481 - val_loss: 1.7508 - val_accuracy: 0.3408
    Epoch 43/50
    18/18 [==============================] - 12s 682ms/step - loss: 1.7591 - accuracy: 0.3400 - val_loss: 1.7973 - val_accuracy: 0.3298
    Epoch 44/50
    18/18 [==============================] - 12s 696ms/step - loss: 1.7245 - accuracy: 0.3477 - val_loss: 1.7573 - val_accuracy: 0.3412
    Epoch 45/50
    18/18 [==============================] - 13s 698ms/step - loss: 1.7119 - accuracy: 0.3494 - val_loss: 1.7202 - val_accuracy: 0.3424
    Epoch 46/50
    18/18 [==============================] - 12s 696ms/step - loss: 1.6755 - accuracy: 0.3592 - val_loss: 1.6835 - val_accuracy: 0.3535
    Epoch 47/50
    18/18 [==============================] - 12s 681ms/step - loss: 1.6471 - accuracy: 0.3676 - val_loss: 1.7367 - val_accuracy: 0.3341
    Epoch 48/50
    18/18 [==============================] - 12s 696ms/step - loss: 1.6417 - accuracy: 0.3675 - val_loss: 1.7396 - val_accuracy: 0.3412
    Epoch 49/50
    18/18 [==============================] - 12s 680ms/step - loss: 1.6576 - accuracy: 0.3599 - val_loss: 1.6600 - val_accuracy: 0.3557
    Epoch 50/50
    18/18 [==============================] - 13s 698ms/step - loss: 1.6218 - accuracy: 0.3722 - val_loss: 1.6145 - val_accuracy: 0.3754
    20/20 [==============================] - 4s 196ms/step - loss: 1.5591 - accuracy: 0.3966
    2/2 [==============================] - 0s 199ms/step - loss: 1.6033 - accuracy: 0.3799
    
    PREDICTION ACCURACY (%):
    Train: 39.660, Test: 37.985



    
![png](output_14_1.png)
    


WEIGHT VISUALIZATION


```python
#attention model
def build_attention_model(attention, batch_size, rnn_units):
    
    # ENCODER STEP
    # ------------
    # Same encoder as before with one and only difference. Now we need all the lstm states so we
    # set return_sequences=True and return_state=True.
    encoder_inputs = Input(shape=(n_timesteps_in, n_features), name='encoder_inputs')
    encoder_lstm = LSTM(rnn_units, return_sequences=True, return_state=True, name='encoder_lstm')
    encoder_outputs, encoder_state_h, encoder_state_c = encoder_lstm(encoder_inputs)
    
    states = [encoder_state_h, encoder_state_c]
    
    # DECODER STEP
    # ------------
    # Set up the decoder layers
    # input shape: (1, n_features + rnn_units)
    decoder_lstm = LSTM(rnn_units, return_state=True, name='decoder_lstm')
    decoder_dense = Dense(n_features, activation='softmax', name='decoder_dense')
    
    # As before, we use as first input the 0-sequence
    all_outputs = []
    inputs = np.zeros((batch_size, 1, n_features))
    
    # Decoder_outputs is the last hidden state of the encoder. Encoder_outputs are all the states
    decoder_outputs = encoder_state_h
    
    # Decoder will only process one time step at a time.
    for _ in range(n_timesteps_in):

        # Pay attention!
        # decoder_outputs (last hidden state) + encoder_outputs (all hidden states)
        context_vector, attention_weights = attention(decoder_outputs, encoder_outputs)
        context_vector = tf.expand_dims(context_vector, 1)

        # create the context vector by applying attention to 
        # Concatenate the input + context vectore to find the next decoder's input
        inputs = tf.concat([context_vector, inputs], axis=-1)

        # Passing the concatenated vector to the LSTM
        # Run the decoder on one timestep with attended input and previous states
        decoder_outputs, state_h, state_c = decoder_lstm(inputs, initial_state=states)        
        outputs = decoder_dense(decoder_outputs)
        
        # Use the last hidden state for prediction the output
        # save the current prediction
        # we will concatenate all predictions later
        outputs = tf.expand_dims(outputs, 1)
        all_outputs.append(outputs)
        
        # Reinject the output (prediction) as inputs for the next loop iteration
        # as well as update the states
        inputs = outputs
        states = [state_h, state_c]
        
    decoder_outputs = Lambda(lambda x: K.concatenate(x, axis=1))(all_outputs)
    model = Model(encoder_inputs, decoder_outputs, name='model_encoder_decoder')
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    
    return model, encoder_lstm, decoder_lstm, decoder_dense
```


```python
def evaluate(seq_in, attention, encoder_lstm, decoder_lstm, decoder_dense):
  attention_plot = np.zeros((n_timesteps_in, n_timesteps_in))

  sequence = one_hot_encode(seq_in,n_features)
  encoder_inputs=np.array(sequence).reshape(1,n_timesteps_in,n_features)
  
  encoder_inputs = tf.convert_to_tensor(encoder_inputs,dtype=tf.float32)

  encoder_outputs, state_h, state_c = encoder_lstm(encoder_inputs)

  states = [state_h, state_c]

  all_outputs = []

  decoder_input_data = np.zeros((1, 1, n_features))
  decoder_input_data[:, 0, 0] = 1 

  inputs = decoder_input_data

  decoder_outputs = state_h

  for t in range(n_timesteps_in):

      context_vector, attention_weights=attention(decoder_outputs, encoder_outputs)

      attention_weights = tf.reshape(attention_weights, (-1, ))
      attention_plot[t] = attention_weights.numpy()

      
      decoder_outputs=tf.expand_dims(decoder_outputs, 1)

      context_vector = tf.expand_dims(context_vector, 1)

      inputs = tf.concat([context_vector, inputs], axis=-1)

      decoder_outputs, state_h, state_c = decoder_lstm(inputs,
                                              initial_state=states)
    
      outputs = decoder_dense(decoder_outputs)

      outputs = tf.expand_dims(outputs, 1)
      all_outputs.append(outputs)

      inputs = outputs
      states = [state_h, state_c]

  decoder_outputs = Lambda(lambda x: K.concatenate(x, axis=1))(all_outputs)
  seq_out=one_hot_decode(decoder_outputs[0])
  
  return seq_in, seq_out, attention_plot
```


```python
from matplotlib import ticker
import seaborn as sns

def plot_attention(attention, sequence, predicted_sequence, weight_n = True):
  """ 
  attention: the attention function used to calc the weight
  if weight_n = True the weight number will appears, else not.
  """
  fig = plt.figure(figsize=(8,8))
  ax = fig.add_subplot(1, 1, 1)
  ax = sns.heatmap(attention, annot=weight_n, \
                     fmt='.2g', cmap='gist_gray')

  fontdict = {'fontsize': 14}

  ax.set_xticklabels([''] + sequence, fontdict=fontdict, rotation=45)
  ax.set_yticklabels([''] + predicted_sequence, rotation = 45, fontdict=fontdict)

  ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
  ax.yaxis.set_major_locator(ticker.MultipleLocator(1))

  plt.show()
```


```python
def translate(seq_in, attention, encoder_lstm, decoder_lstm, decoder_dense):
  seq_in, seq_out, attention_plot = evaluate(seq_in, attention, encoder_lstm, decoder_lstm, decoder_dense)

  print('Input: %s' % (seq_in))
  print('Predicted translation: {}'.format(seq_out))

  attention_plot = attention_plot[:len(seq_out), :len(seq_in)]
  plot_attention(attention_plot, seq_in, seq_out)
```


```python
# dataset 
n_timesteps_in = 10  # Sequence length
n_features = 20     # Maximum number allowed-1 (length of the one-hot-encoded vector)
train_size = 2000 
test_size = 200
X_train, y_train, X_test, y_test = create_dataset(train_size, test_size, n_timesteps_in,n_features)

# training parameters
batch_size = 100

# model parameters
rnn_units = 100
```


```python
test = [random.randint(0,n_features) for i in range(n_timesteps_in)]
```


```python
# attention model
attention = LuongDotAttention()
model_attention, encoder_lstm, decoder_lstm, decoder_dense  = build_attention_model(attention, batch_size, rnn_units)
```


```python
train_test(model_attention, X_train, y_train , X_test,
           y_test, batch_size=batch_size,epochs=50, patience=3)
```

    Epoch 1/50
    18/18 [==============================] - 1s 47ms/step - loss: 1.8149 - accuracy: 0.3387 - val_loss: 1.8185 - val_accuracy: 0.3335
    Epoch 2/50
    18/18 [==============================] - 1s 48ms/step - loss: 1.7003 - accuracy: 0.3657 - val_loss: 1.6938 - val_accuracy: 0.3720
    Epoch 3/50
    18/18 [==============================] - 1s 48ms/step - loss: 1.5632 - accuracy: 0.4119 - val_loss: 1.5334 - val_accuracy: 0.4385
    Epoch 4/50
    18/18 [==============================] - 1s 46ms/step - loss: 1.4016 - accuracy: 0.4823 - val_loss: 1.3537 - val_accuracy: 0.5100
    Epoch 5/50
    18/18 [==============================] - 1s 50ms/step - loss: 1.2218 - accuracy: 0.5621 - val_loss: 1.1728 - val_accuracy: 0.5915
    Epoch 6/50
    18/18 [==============================] - 1s 47ms/step - loss: 1.0462 - accuracy: 0.6467 - val_loss: 0.9783 - val_accuracy: 0.6825
    Epoch 7/50
    18/18 [==============================] - 1s 49ms/step - loss: 0.8640 - accuracy: 0.7362 - val_loss: 0.8102 - val_accuracy: 0.7550
    Epoch 8/50
    18/18 [==============================] - 1s 49ms/step - loss: 0.6932 - accuracy: 0.8134 - val_loss: 0.6319 - val_accuracy: 0.8325
    Epoch 9/50
    18/18 [==============================] - 1s 53ms/step - loss: 0.5353 - accuracy: 0.8779 - val_loss: 0.4867 - val_accuracy: 0.8915
    Epoch 10/50
    18/18 [==============================] - 1s 49ms/step - loss: 0.4010 - accuracy: 0.9272 - val_loss: 0.3608 - val_accuracy: 0.9300
    Epoch 11/50
    18/18 [==============================] - 1s 52ms/step - loss: 0.2942 - accuracy: 0.9586 - val_loss: 0.2695 - val_accuracy: 0.9590
    Epoch 12/50
    18/18 [==============================] - 1s 46ms/step - loss: 0.2213 - accuracy: 0.9728 - val_loss: 0.1965 - val_accuracy: 0.9785
    Epoch 13/50
    18/18 [==============================] - 1s 51ms/step - loss: 0.1617 - accuracy: 0.9830 - val_loss: 0.1513 - val_accuracy: 0.9870
    Epoch 14/50
    18/18 [==============================] - 1s 47ms/step - loss: 0.1215 - accuracy: 0.9895 - val_loss: 0.1160 - val_accuracy: 0.9895
    Epoch 15/50
    18/18 [==============================] - 1s 48ms/step - loss: 0.0953 - accuracy: 0.9927 - val_loss: 0.0974 - val_accuracy: 0.9915
    Epoch 16/50
    18/18 [==============================] - 1s 51ms/step - loss: 0.0775 - accuracy: 0.9952 - val_loss: 0.0763 - val_accuracy: 0.9940
    Epoch 17/50
    18/18 [==============================] - 1s 50ms/step - loss: 0.0635 - accuracy: 0.9970 - val_loss: 0.0665 - val_accuracy: 0.9940
    Epoch 18/50
    18/18 [==============================] - 1s 48ms/step - loss: 0.0534 - accuracy: 0.9986 - val_loss: 0.0577 - val_accuracy: 0.9940
    Epoch 19/50
    18/18 [==============================] - 1s 50ms/step - loss: 0.0455 - accuracy: 0.9989 - val_loss: 0.0506 - val_accuracy: 0.9960
    Epoch 20/50
    18/18 [==============================] - 1s 51ms/step - loss: 0.0398 - accuracy: 0.9997 - val_loss: 0.0439 - val_accuracy: 0.9960
    Epoch 21/50
    18/18 [==============================] - 1s 50ms/step - loss: 0.0347 - accuracy: 0.9999 - val_loss: 0.0415 - val_accuracy: 0.9960
    Epoch 22/50
    18/18 [==============================] - 1s 50ms/step - loss: 0.0308 - accuracy: 0.9999 - val_loss: 0.0367 - val_accuracy: 0.9970
    Epoch 23/50
    18/18 [==============================] - 1s 50ms/step - loss: 0.0278 - accuracy: 0.9999 - val_loss: 0.0358 - val_accuracy: 0.9970
    Epoch 24/50
    18/18 [==============================] - 1s 50ms/step - loss: 0.0251 - accuracy: 1.0000 - val_loss: 0.0305 - val_accuracy: 0.9975
    Epoch 25/50
    18/18 [==============================] - 1s 49ms/step - loss: 0.0229 - accuracy: 1.0000 - val_loss: 0.0277 - val_accuracy: 0.9980
    Epoch 26/50
    18/18 [==============================] - 1s 52ms/step - loss: 0.0208 - accuracy: 1.0000 - val_loss: 0.0260 - val_accuracy: 0.9980
    Epoch 27/50
    18/18 [==============================] - 1s 54ms/step - loss: 0.0190 - accuracy: 1.0000 - val_loss: 0.0245 - val_accuracy: 0.9980
    Epoch 28/50
    18/18 [==============================] - 1s 50ms/step - loss: 0.0175 - accuracy: 1.0000 - val_loss: 0.0227 - val_accuracy: 0.9980
    Epoch 29/50
    18/18 [==============================] - 1s 45ms/step - loss: 0.0162 - accuracy: 1.0000 - val_loss: 0.0214 - val_accuracy: 0.9985
    Epoch 30/50
    18/18 [==============================] - 1s 46ms/step - loss: 0.0151 - accuracy: 1.0000 - val_loss: 0.0206 - val_accuracy: 0.9990
    Epoch 31/50
    18/18 [==============================] - 1s 50ms/step - loss: 0.0141 - accuracy: 1.0000 - val_loss: 0.0195 - val_accuracy: 0.9985
    Epoch 32/50
    18/18 [==============================] - 1s 48ms/step - loss: 0.0132 - accuracy: 1.0000 - val_loss: 0.0185 - val_accuracy: 0.9985
    Epoch 33/50
    18/18 [==============================] - 1s 50ms/step - loss: 0.0123 - accuracy: 1.0000 - val_loss: 0.0180 - val_accuracy: 0.9990
    Epoch 34/50
    18/18 [==============================] - 1s 49ms/step - loss: 0.0116 - accuracy: 1.0000 - val_loss: 0.0171 - val_accuracy: 0.9985
    Epoch 35/50
    18/18 [==============================] - 1s 53ms/step - loss: 0.0109 - accuracy: 1.0000 - val_loss: 0.0161 - val_accuracy: 0.9985
    Epoch 36/50
    18/18 [==============================] - 1s 45ms/step - loss: 0.0103 - accuracy: 1.0000 - val_loss: 0.0157 - val_accuracy: 0.9990
    Epoch 37/50
    18/18 [==============================] - 1s 52ms/step - loss: 0.0098 - accuracy: 1.0000 - val_loss: 0.0154 - val_accuracy: 0.9990
    Epoch 38/50
    18/18 [==============================] - 1s 50ms/step - loss: 0.0093 - accuracy: 1.0000 - val_loss: 0.0152 - val_accuracy: 0.9990
    Epoch 39/50
    18/18 [==============================] - 1s 53ms/step - loss: 0.0088 - accuracy: 1.0000 - val_loss: 0.0143 - val_accuracy: 0.9990
    Epoch 40/50
    18/18 [==============================] - 1s 47ms/step - loss: 0.0083 - accuracy: 1.0000 - val_loss: 0.0138 - val_accuracy: 0.9990
    Epoch 41/50
    18/18 [==============================] - 1s 48ms/step - loss: 0.0079 - accuracy: 1.0000 - val_loss: 0.0131 - val_accuracy: 0.9990
    Epoch 42/50
    18/18 [==============================] - 1s 49ms/step - loss: 0.0076 - accuracy: 1.0000 - val_loss: 0.0123 - val_accuracy: 0.9990
    Epoch 43/50
    18/18 [==============================] - 1s 51ms/step - loss: 0.0072 - accuracy: 1.0000 - val_loss: 0.0124 - val_accuracy: 0.9990
    Epoch 44/50
    18/18 [==============================] - 1s 46ms/step - loss: 0.0069 - accuracy: 1.0000 - val_loss: 0.0118 - val_accuracy: 0.9990
    Epoch 45/50
    18/18 [==============================] - 1s 46ms/step - loss: 0.0066 - accuracy: 1.0000 - val_loss: 0.0118 - val_accuracy: 0.9990
    Epoch 46/50
    18/18 [==============================] - 1s 44ms/step - loss: 0.0063 - accuracy: 1.0000 - val_loss: 0.0112 - val_accuracy: 0.9990
    Epoch 47/50
    18/18 [==============================] - 1s 51ms/step - loss: 0.0060 - accuracy: 1.0000 - val_loss: 0.0112 - val_accuracy: 0.9990
    Epoch 48/50
    18/18 [==============================] - 1s 54ms/step - loss: 0.0058 - accuracy: 1.0000 - val_loss: 0.0109 - val_accuracy: 0.9990
    Epoch 49/50
    18/18 [==============================] - 1s 51ms/step - loss: 0.0056 - accuracy: 1.0000 - val_loss: 0.0105 - val_accuracy: 0.9990
    Epoch 50/50
    18/18 [==============================] - 1s 50ms/step - loss: 0.0053 - accuracy: 1.0000 - val_loss: 0.0103 - val_accuracy: 0.9990
    20/20 [==============================] - 0s 15ms/step - loss: 0.0057 - accuracy: 0.9999
    2/2 [==============================] - 0s 21ms/step - loss: 0.0076 - accuracy: 1.0000
    
    PREDICTION ACCURACY (%):
    Train: 99.990, Test: 100.000



    
![png](output_23_1.png)
    



```python
translate(test, attention, encoder_lstm, decoder_lstm, decoder_dense)
```

    Input: [18, 11, 13, 0, 7, 9, 19, 5, 2, 9]
    Predicted translation: [9, 2, 5, 19, 9, 7, 7, 13, 11, 18]



    
![png](output_24_1.png)
    


# Report

+ Bahdanau Attention and Luong General Attention implementation.
+ Comparative.
+ Weight visualization. 
+ Blog site.


