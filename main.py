# %%
import time

import numpy as np
from IPython.display import Image
from keras.callbacks import TensorBoard
from keras.layers import Dense
from keras.models import Sequential
from pyswarms.single.global_best import GlobalBestPSO
from pyswarms.utils.functions import single_obj as fx
from pyswarms.utils.plotters import plot_surface
from pyswarms.utils.plotters.formatters import Animator, Designer, Mesher
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# Data set transformation

iris = load_iris()
X = iris['data']
y = iris['target']
names = iris['target_names']
feature_names = iris['feature_names']
enc = OneHotEncoder()
Y = enc.fit_transform(y[:, np.newaxis]).toarray()
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_train, X_test, Y_train, Y_test = train_test_split(
    X_scaled, Y, test_size=0.5, random_state=2)
n_features = X.shape[1]
n_classes = Y.shape[1]

## Building the neural network


def create_custom_model(input_dim, output_dim, nodes, n=1, name='model'):
    model = Sequential(name=name)
    for i in range(n):
        model.add(Dense(nodes, input_dim=input_dim, activation='relu'))
    model.add(Dense(output_dim, activation='softmax'))
    model.compile(loss='categorical_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])
    return model


n_layers = 1
model = create_custom_model(n_features, n_classes,
                            10, n_layers)
model.summary()

start_time = time.time()
print('Model name:', model.name)
history_callback = model.fit(X_train, Y_train,
                             batch_size=5,
                             epochs=300,
                             validation_data=(X_test, Y_test)
                             )
score = model.evaluate(X_test, Y_test)
print('Test loss:', score[0])
print('Test accuracy:', score[1])
print("--- %s seconds ---" % (time.time() - start_time))
#%%
## Building the PSO function and optimization
def get_shape(model):
    weights_layer = model.get_weights()
    shapes = []
    for weights in weights_layer:
        shapes.append(weights.shape)
    return shapes
def set_shape(weights,shapes):
    new_weights = []
    index=0
    for shape in shapes:
        if(len(shape)>1):
            n_nodes = np.prod(shape)+index
        else:
            n_nodes=shape[0]+index
        tmp = np.array(weights[index:n_nodes]).reshape(shape)
        new_weights.append(tmp)
        index=n_nodes
    return new_weights

#%%
start_time = time.time()
def evaluate_nn(W, shape,X_test=X_train, Y_test=Y_train):
    result = []
    for weights in W:
        model.set_weights(set_shape(weights,shape))
        score = model.evaluate(X_test, Y_test, verbose=0)
        result.append(1-score[1])
    return result

shape = get_shape(model)
x_max = 1.0 * np.ones(83)
x_min = -1.0 * x_max
bounds = (x_min, x_max)
options = {'c1': 0.3, 'c2': 0.8, 'w': 0.4}
optimizer = GlobalBestPSO(n_particles=40, dimensions=83,
                          options=options, bounds=bounds)
cost, pos = optimizer.optimize(evaluate_nn, 10, X_test=X_train, Y_test=Y_train,shape=shape)
model.set_weights(set_shape(pos,shape))
score = model.evaluate(X_test, Y_test)
print('Test loss:', score[0])
print('Test accuracy:', score[1])
print("--- %s seconds ---" % (time.time() - start_time))
# %%
# Use sphere function

m = Mesher(func=fx.sphere)
pos_history = [pos[:, :2] for pos in optimizer.pos_history]
pos3d = m.compute_history_3d(pos_history)
# Assuming we already had an optimizer ready
my_animator = Animator(repeat=False)
my_designer = Designer(figsize=(6, 6))
animation = plot_surface(pos3d, animator=my_animator, designer=my_designer)
# %%
animation.save('pso.gif', writer='imagemagick', fps=6, )
Image(url='pso.gif')
# %%
