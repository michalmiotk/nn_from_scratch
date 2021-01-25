import numpy as np

def relu(some_input):
    return np.where(some_input>0, some_input, 0)

def relu_deriv(some_input):
    return np.where(some_input>0, some_input, 0)

def sigmoid(some_input):
    return 1/(1+np.exp(-some_input))

def sigmoid_deriv(some_input):
    return np.multiply(sigmoid(some_input), 1-sigmoid(some_input))

def get_deriv(activation_name):
    if activation_name == "relu":
        return relu_deriv
    if activation_name == "sigmoid":
        return sigmoid_deriv

def get_activation(activation_name):
    if activation_name == "relu":
        return relu
    if activation_name == "sigmoid":
        return sigmoid

nn_structure = [{"in_dim":3, "out_dim":4, "activation":"relu"},
                {"in_dim":4, "out_dim":5, "activation":"relu"}, 
                {"in_dim":5, "out_dim":1, "activation":"sigmoid"},
]

weights = { "w0":0.1*np.random.rand(nn_structure[0]["in_dim"], nn_structure[0]["out_dim"]),
            "b0":0.1*np.random.rand(nn_structure[0]["out_dim"]),
            "w1":0.1*np.random.rand(nn_structure[1]["in_dim"], nn_structure[1]["out_dim"]),
            "b1":0.1*np.random.rand(nn_structure[1]["out_dim"]),
            "w2":0.1*np.random.rand(nn_structure[2]["in_dim"], nn_structure[2]["out_dim"]),
            "b2":0.1*np.random.rand(nn_structure[2]["out_dim"]),
}

def forward(input):
    memory = {}
    assert input.shape[1] == nn_structure[0]["in_dim"]
    x_z = np.dot(input,weights["w0"])+input.shape[0]*weights["b0"]
    x_a = get_activation(nn_structure[0]["activation"])(x_z)
    memory['0i'] = input
    memory['0z'] = x_z
    memory['0a'] = x_a
    for i in range(1,len(nn_structure)):
        memory[str(i)+'i'] = x_a
        x_z = np.dot(x_a,weights["w"+str(i)])+x_a.shape[0]*weights["b"+str(i)]
        x_a = get_activation(nn_structure[i]["activation"])(x_z)
        memory[str(i)+'z'] = x_z
        memory[str(i)+'a'] = x_a
    return x_a, memory

def backward(tar, memory_forward,out):
    memory_backward = {}
    loss = (out-tar)**2
    for i in range(len(nn_structure)-1,-1, -1):
        if i == len(nn_structure)-1:
            out = np.mean(out)
            derror = 2*(out - tar) #must be later_filled
        else:
            derror = memory_backward[str(i+1)+'en']
            derror = np.mean(derror, 1, keepdims=True)
        dActivation_over_out = get_deriv(nn_structure[i]['activation'])(memory_forward[str(i)+'z'])
        dout_dW = memory_forward[str(i)+'i']
        #https://www.geeksforgeeks.org/implementation-of-neural-network-from-scratch-using-numpy/
        dout_db = np.expand_dims(np.ones(weights['b'+str(i)].shape),0)
        dLoss_dout = np.dot(derror, dActivation_over_out)
        

        dLoss_dW = np.dot(np.transpose(dout_dW),dLoss_dout)
        dLoss_db = np.dot(np.transpose(dout_db), dLoss_dout).mean(0)
        memory_backward['dW'+str(i)] = dLoss_dW
        memory_backward['db'+str(i)] = dLoss_db
        memory_backward[str(i)+'en'] = dActivation_over_out
    return memory_backward, loss

epochs_nr=50
lr = 0.19
random_input = np.random.rand(1,3)
for i in range(epochs_nr):
    out, memory_forward = forward(random_input)
    memory_backward,loss = backward(1, memory_forward, out)
    for key in weights:
        if key.startswith('w'):
            weights[key] -= lr*memory_backward['dW' + key.lstrip('w')]
        if key.startswith('b'):
            weights[key] -= lr*memory_backward['db' + key.lstrip('b')]

    print("loss", loss)