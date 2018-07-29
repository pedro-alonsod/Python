# graph model with one input and two outputs
graph = Graph()
graph.add_input(name='input', input_shape=(32,))
graph.add_node(Dense(16), name='dense1', input='input')
graph.add_node(Dense(4), name='dense2', input='input')
graph.add_node(Dense(4), name='dense3', input='dense1')
graph.add_output(name='output1', input='dense2')
graph.add_output(name='output2', input='dense3')
graph.compile(optimizer='rmsprop', loss={'output1':'mse',
'output2':'mse'})
history = graph.fit({'input':X_train, 'output1':y_train,
'output2':y2_train}, nb_epoch=10)