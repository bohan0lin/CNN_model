import numpy as np
# import matplotlib.pyplot as plt
# import scipy
import cnn_utils as U
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable

torch.manual_seed(0)

# model train
def model_train(rnn, train_data_x, train_data_y, test_data_x, test_data_y, epoch):
    
    rnn.train()   
    tr_loss = 0
    train_data_x, train_data_y = Variable(train_data_x), Variable(train_data_y)
    test_data_x, test_data_y = Variable(test_data_x), Variable(test_data_y)

    optimizer.zero_grad()
    
    train_data_x = train_data_x.to(dtype=torch.float)
    test_data_x = test_data_x.to(dtype=torch.float)

    # print("train_y", train_data_y.shape)
    # print("test_y", test_data_y.shape)

    output_train= rnn(train_data_x)
    output_test= rnn(test_data_x)

    # print("train_output", output_train.shape)
    # print("test_output", output_test.shape)

    train_data_y = train_data_y.squeeze()
    test_data_y = test_data_y.squeeze()

    loss_train = loss_function(output_train, train_data_y)
    loss_val = loss_function(output_test, test_data_y)

#     train_losses.append(loss_train)
#     test_losses.append(loss_val)
    
    loss_train.backward()
    optimizer.step()
    
    tr_loss = loss_train.item()
    
    if epoch%10 == 0:
        # printing the validation loss
        print('Epoch : ',epoch, '\t', 'loss :', loss_val)

if __name__ == '__main__':
    # load datasets
    train_data_x, train_data_y, test_data_x, test_data_y = load_dataset()
    # train_data_x: (205, 64, 64, 3)
    train_data_x = train_data_x.transpose(0, 3, 1, 2)
    test_data_x = test_data_x.transpose(0, 3, 1, 2)
    
    # rescale data 
    train_data_x = train_data_x / 255.0
    test_data_x = test_data_x / 255.0

   
    batch_size = 5
    learning_rate = 0.0005
    epoch = 120
    

#     train_losses = []
#     test_losses = []
    
    train_data_x  = torch.from_numpy(train_data_x)
    train_data_y  = torch.from_numpy(train_data_y)

    test_data_x  = torch.from_numpy(test_data_x)
    test_data_y  = torch.from_numpy(test_data_y)

    rnn = Net()
    optimizer = optim.Adam(rnn.parameters(), lr=learning_rate)
    loss_function = nn.CrossEntropyLoss()
    
    # model train (model test function can be called directly in model_train)
    for epoch in range(epoch):
        model_train(rnn, train_data_x, train_data_y, test_data_x, test_data_y, epoch)


    # Train data accuracy
    train_data_x = train_data_x.to(dtype=torch.float)
    test_data_x = test_data_x.to(dtype=torch.float)
    train_data_y = train_data_y.to(dtype=torch.float)
    test_data_y = test_data_y.to(dtype=torch.float)

    rnn.eval()
    with torch.no_grad():
        output = rnn(train_data_x)
    
    softmax = torch.exp(output)
    prob = list(softmax.numpy())
    y_train_predictions = np.argmax(prob, axis=1)
    
    # train_data_y = train_data_y.numpy()
    # print("train_data_y", train_data_y.shape)
    # print("predictions", predictions)
    
    # accuracy on training set
    print("Train_accuracy:", accuracy_score(train_data_y.T, y_train_predictions))


    # Test data accuracy
    with torch.no_grad():
        output = rnn(test_data_x)
    
    softmax = torch.exp(output)
    prob = list(softmax.numpy())
    y_test_predictions = np.argmax(prob, axis=1)
    
    # test_data_y = test_data_y.numpy()
    # print("test_data_y", train_data_y)
    # print("predictions", y_test_predictions)
    
    # accuracy on training set
#     print('y_test_predictions:', y_test_predictions)
    print("Test_accuracy:", accuracy_score(test_data_y.T, y_test_predictions))