import numpy as np
import pandas as pd
import tqdm
from data_utils import load_dataset
import matplotlib.pyplot as plt
from sklearn import neighbors
import time

__author__ = 'En Xu Li (Thomas)'
__date__ = 'February 9, 2020'


def _partition_fold(v,data):
    """
    partition the data ready for cross validation

    Inputs:
        v: (int) cross validation parameter, number of cross folds
        data: (np.array) training data

    Outputs:
        list of partitioned indicies
    """

    partition = []
    for i in range(v):
        if i==v-1:
            partition.append(range(int(i*len(data)/5),len(data)))
        else:
            partition.append(range(int(i*len(data)/5),int(i*len(data)/5+(len(data)/5))))
    return partition

def _l1_dist(x,y):
    """
    l1 distance metric

    Inputs:
        x,y: (np.array)

    Outputs:
        l1 difference (scalar)
    """
    return np.sum(abs(x-y))

def _l2_dist(x,y):
    """
    l2 distance metric

    Inputs:
        x,y: (np.array)

    Outputs:
        l2 difference (scalar)
    """
    return np.linalg.norm([x-y],ord=2)

def _RMSE(x,y):
    """
    Root Mean Square Error

    Inputs:
        x,y: (np.array)

    Outputs:
        Root Mean Square difference of x and y
    """
    return np.sqrt(np.mean((x-y)**2))

def _cast_TF(x):
    """
    change bool type array to one hot encoding with 1 and 0

    Inputs:
        x: (bool type np.array)

    Outputs:
        numpy array with one hot encoding
    """
    return np.where(x==True,1,0)

def _eval_knn(k,train_x,train_y,query_x,query_y,dist_metric,compute_loss=True):
    """
    knn algorithm

    Inputs:
        k: (list) k[0]:lower bound of number of nearest neighbours; k[1]:upper bound of number of nearest neighbours
        train_x: (np.array) input training vector
        train_y: (np.array) input target vector
        query_x: (np.array) input query vector
        query_y: (np.array) input query target ground truth vector
        dist_metric: (str) 'l1' or 'l2'
        compute_loss: (bool)

    Outputs:
        RMSE of predicted if compute_loss True
        predicted vector on query_x if compute_loss False
    """
    neighbours = {}
    predicted = {}
    rval_loss = np.empty((0,1))
    for j in range(k[0],k[1]):
        predicted['k='+str(j)] = np.empty((0,train_y.shape[-1]))
    bar = tqdm.tqdm(total=query_x.shape[0], desc='Query', position=0)
    for pt in query_x:
        bar.update(1)
        distances = []
        for idx in range(train_x.shape[0]):
            data = train_x[idx]
            if dist_metric=='l1': dist = _l1_dist(pt, data)
            elif dist_metric=='l2': dist = _l2_dist(pt,data)
            else: assert 'distance metric invalid'
            distances.append((idx, dist))
        distances.sort(key=lambda tup: tup[1])

        for j in range(k[0],k[1]):
            neighbours['k='+str(j)] = np.empty((0,train_y.shape[-1]))
            for i in range(j):
                neighbours['k='+str(j)] = np.append(neighbours['k='+str(j)], [train_y[distances[i][0]]], axis=0)
            predicted['k='+str(j)] = np.append(predicted['k='+str(j)], [neighbours['k='+str(j)].mean(axis=0)], axis=0)

    if compute_loss:
        for k in predicted:
            rval_loss = np.append(rval_loss,[_RMSE(predicted[k],query_y)])
        return rval_loss
    else:

        return predicted

def _cross_val(dataset='mauna_loa',k=10,dist_metric='l1',v=5):
    """
    cross validation technique on knn

    Inputs:
        dataset: (str) name of dataset
        k: (list) k[0]:lower bound of number of nearest neighbours; k[1]:upper bound of number of nearest neighbours
        dist_metric: (str) 'l1' or 'l2'
        v: (int) cross validation parameter, number of cross folds

    Outputs:
        averaged validation loss
    """
    print ('------Processing Dataset '+dataset+' ------')
    if dataset=='rosenbrock':
        x_train, x_valid, x_test, y_train, y_valid, y_test = load_dataset('rosenbrock', n_train=5000, d=2)
    else:
        x_train, x_valid, x_test, y_train, y_valid, y_test = load_dataset(dataset)

    x_train = np.vstack([x_valid, x_train])
    y_train = np.vstack([y_valid, y_train])

    np.random.seed(42)
    np.random.shuffle(x_train)
    np.random.seed(42)
    np.random.shuffle(y_train)

    data_partition = _partition_fold(v=v,data=x_train)
    loss = np.empty((0,k[1]-k[0]))
    for fold in range(v):
        print ('------Processing Fold '+str(fold+1)+' ------')
        train_x = np.delete(x_train, list(data_partition[fold]), axis=0)
        train_y = np.delete(y_train, list(data_partition[fold]), axis=0)

        query_x = np.take(x_train,list(data_partition[fold]),axis=0)
        query_y = np.take(y_train,list(data_partition[fold]),axis=0)

        curr_loss = _eval_knn(k,train_x,train_y,query_x,query_y,dist_metric=dist_metric)
        loss = np.append(loss,[curr_loss],axis=0)

    loss = loss.mean(axis=0)
    return loss

def _classification(dataset='iris',k_range=[1,31],dist_metric='l1'):
    """
    knn on classificaiton dataset

    Inputs:
        dataset: (str) name of dataset
        k: (list) k[0]:lower bound of number of nearest neighbours; k[1]:upper bound of number of nearest neighbours
        dist_metric: (str) 'l1' or 'l2'

    Outputs:
        validation accuracy
    """
    print ('------Processing Dataset '+dataset+' ------')

    x_train, x_valid, x_test, y_train, y_valid, y_test = load_dataset(dataset)

    if y_train.dtype==np.dtype('bool'):
        y_train = _cast_TF(y_train)
        y_valid = _cast_TF(y_valid)
        y_test = _cast_TF(y_test)
    acc = []
    predicted = _eval_knn(k_range,x_train,y_train,x_valid,y_valid,dist_metric,compute_loss=False)
    for k in range(k_range[0],k_range[1]):
        #print(k)
        curr_predict = predicted['k='+str(k)]
        #print(curr_predict)
        result = np.argmax(curr_predict,axis=1)
        #print(result)
        gt = np.where(y_valid==True,1,0)
        gt = np.argmax(gt,axis=1)

        unique, counts = np.unique(result-gt, return_counts=True)
        correct = dict(zip(unique, counts))[0]
        #print(correct)
        acc.append(correct/y_valid.shape[0])

    return acc

def _test_regression(dataset='mauna_loa',k=1,dist_metric='l2',d=2):
    """
    compute test loss on regression dataset

    Inputs:
        dataset: (str) name of dataset
        k: (int) number of nearest neighbours to test on
        dist_metric: (str) 'l1' or 'l2'
        d : (int, optional) if name='rosenbrock' the specify the dataset dimensionality

    Outputs:
        RMSE on test set of the dataset
    """

    if dataset=='rosenbrock':
        x_train, x_valid, x_test, y_train, y_valid, y_test = load_dataset('rosenbrock', n_train=5000, d=d)
    else:
        x_train, x_valid, x_test, y_train, y_valid, y_test = load_dataset(dataset)

    x_train = np.vstack([x_valid, x_train])
    y_train = np.vstack([y_valid, y_train])
    return _eval_knn([k,k+1],x_train,y_train,x_test,y_test,dist_metric,compute_loss=True)

def predict_cross_val(dataset='mauna_loa',k=2,dist_metric='l2',v=5):
    """
    cross validation technique on knn and output predicted values

    Inputs:
        dataset: (str) name of dataset
        k: (list) k[0]:lower bound of number of nearest neighbours; k[1]:upper bound of number of nearest neighbours
        dist_metric: (str) 'l1' or 'l2'
        v: (int) cross validation parameter, number of cross folds

    Outputs:
        [predict_x,GroundTruth_y,predicted_y]
    """
    x_train, x_valid, x_test, y_train, y_valid, y_test = load_dataset(dataset)

    x_train = np.vstack([x_valid, x_train])
    y_train = np.vstack([y_valid, y_train])

    np.random.seed(42)
    np.random.shuffle(x_train)
    np.random.seed(42)
    np.random.shuffle(y_train)


    data_partition = _partition_fold(v=v,data=x_train)
    predicted_y = np.empty((0,y_train.shape[-1]))
    for fold in range(v):
        print ('------Processing Fold '+str(fold+1)+' ------')
        train_x = np.delete(x_train, data_partition[fold], axis=0)
        train_y = np.delete(y_train, data_partition[fold], axis=0)
        query_x = np.take(x_train,data_partition[fold],axis=0)
        query_y = np.take(y_train,data_partition[fold],axis=0)

        curr_predict = _eval_knn([k,k+1],train_x,train_y,query_x,query_y,dist_metric=dist_metric,compute_loss=False)
        #print(curr_predict.shape)
        predicted_y = np.append(predicted_y,curr_predict['k='+str(k)],axis=0)

    rval = []
    for idx in range(x_train.shape[0]):
        rval.append((x_train[idx],y_train[idx],predicted_y[idx]))

    rval.sort(key=lambda tup: tup[0])
    return [i[0] for i in rval],[i[1] for i in rval],[i[2] for i in rval]

def run_Q1(k_range=[1,30]):
    """
    script to do cross validation on regression dataset

    Inputs:
        k_range: (list) k[0]:lower bound of number of nearest neighbours; k[1]:upper bound of number of nearest neighbours

    Outputs:
        None
    """
    output_data = {}
    index = []
    for k in range(k_range[0],k_range[1]):
        index += ['k='+str(k)]
    data_list = ['mauna_loa','rosenbrock','pumadyn32nm']

    print('L2 loss')
    for data in data_list:

        val = _cross_val(dataset=data,k=k_range,dist_metric='l2',v=5)
        print(val)
        output_data[data] = val

    df = pd.DataFrame(output_data, index =index)
    with pd.option_context('display.max_rows', None, 'display.max_columns', None):  # more options can be specified also
        print(df)
    df.to_csv ('l2.csv')
    #     #print (loss)
    output_data = {}
    print('L1 loss')
    for data in data_list:

        val = _cross_val(dataset=data,k=k_range,dist_metric='l1',v=5)
        output_data[data] = val
        print(output_data)

    df = pd.DataFrame(output_data, index =index)
    with pd.option_context('display.max_rows', None, 'display.max_columns', None):  # more options can be specified also
        print(df)
    df.to_csv ('l1.csv')

def predict_test(dataset='mauna_loa',k=2,dist_metric='l2'):
    """
    run knn and output predicted values on regression test data

    Inputs:
        dataset: (str) name of dataset
        k: (list) k[0]:lower bound of number of nearest neighbours; k[1]:upper bound of number of nearest neighbours
        dist_metric: (str) 'l1' or 'l2'

    Outputs:
        [predict_x,GroundTruth_y,predicted_y]
    """
    x_train, x_valid, x_test, y_train, y_valid, y_test = load_dataset(dataset)

    x_train = np.vstack([x_valid, x_train])
    y_train = np.vstack([y_valid, y_train])


    predicted_y = np.empty((0,y_train.shape[-1]))
    curr_predict = _eval_knn([k,k+1],x_train,y_train,x_test,y_test,dist_metric=dist_metric,compute_loss=False)
    predicted_y = np.append(predicted_y,curr_predict['k='+str(k)],axis=0)

    rval = []
    for idx in range(x_test.shape[0]):
        rval.append((x_test[idx],y_test[idx],predicted_y[idx]))

    rval.sort(key=lambda tup: tup[0])
    return [i[0] for i in rval],[i[1] for i in rval],[i[2] for i in rval]

def _test_classification(dataset='iris',k_range=[1,2],dist_metric='l1'):
    """
    run knn and output predicted values on classificaiton test data

    Inputs:
        dataset: (str) name of dataset
        k_range: (list) k[0]:lower bound of number of nearest neighbours; k[1]:upper bound of number of nearest neighbours
        dist_metric: (str) 'l1' or 'l2'


    Outputs:
        accuracy of predicted values referred to GroundTruth
    """

    print ('------Processing Dataset '+dataset+' ------')

    x_train, x_valid, x_test, y_train, y_valid, y_test = load_dataset(dataset)

    if y_train.dtype==np.dtype('bool'):
        y_train = _cast_TF(y_train)
        y_valid = _cast_TF(y_valid)
        y_test = _cast_TF(y_test)
    acc = []
    predicted = _eval_knn(k_range,x_train,y_train,x_test,y_test,dist_metric,compute_loss=False)
    for k in range(k_range[0],k_range[1]):
        curr_predict = predicted['k='+str(k)]
        result = np.argmax(curr_predict,axis=1)
        gt = np.where(y_test==True,1,0)
        gt = np.argmax(gt,axis=1)
        #print(result-gt)
        #break

        unique, counts = np.unique(result-gt, return_counts=True)
        correct = dict(zip(unique, counts))[0]
        acc.append(correct/y_test.shape[0])

    return acc

def run_Q2(k_range=[1,31]):
    """
    script to run knn on classificaiton dataset

    Inputs:
        k_range: (list) k[0]:lower bound of number of nearest neighbours; k[1]:upper bound of number of nearest neighbours

    Outputs:
        None
    """
    output_data = {}
    index = []
    for k in range(k_range[0],k_range[1]):
        index += ['k='+str(k)]
    data_list = ['iris']
    #data_list = ['mnist_small']

    print('L2 loss')
    for data in data_list:
        #for k in range(k_range[0],k_range[1]):
        #    print ('------Processing k = '+str(k)+' ------')
        acc = _classification(dataset=data,k_range=k_range,dist_metric='l2')
        print(acc)
        output_data[data] = acc

    df = pd.DataFrame(output_data, index =index)
    with pd.option_context('display.max_rows', None, 'display.max_columns', None):  # more options can be specified also
        print(df)
    df.to_csv ('q2_l2.csv')
    #     #print (loss)
    output_data = {}
    print('L1 loss')
    for data in data_list:
        #for k in range(k_range[0],k_range[1]):
        #    print ('------Processing k = '+str(k)+' ------')
        acc = _classification(dataset=data,k_range=k_range,dist_metric='l1')
        print(acc)
        output_data[data] = acc

    df = pd.DataFrame(output_data, index =index)
    with pd.option_context('display.max_rows', None, 'display.max_columns', None):  # more options can be specified also
        print(df)
    df.to_csv ('q2_l1.csv')

def plot(xlabel='',ylabel='',name='fig',x=None,y=None,legend=None):
    """
    plot and figures

    Inputs:
        xlabel: (str) label on x axis
        ylabel: (str) label on y axis
        name: (str) title of the figure
        x: (np.array) x data
        y: (list of np.array) list of y values to plot against x
        legend: (list of str) label on y values

    Outputs:
        None
    """
    fig = plt.figure()
    for i in range(len(y)):
        plt.plot(x,y[i],label=legend[i])
    plt.legend()
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(name)
    fig.savefig(name+'.png')

def _kd_tree(dataset='rosenbrock',dist_metric='l2',k=5,d=2):
    """
    knn using kd_tree

    Inputs:
        dataset: (str) name of dataset
        k: (int) number of nearest neighbours
        dist_metric: (str) 'l1' or 'l2'
        d: (int) data dimensionality

    Outputs:
        RMSE on predicted values
    """
    x_train, x_valid, x_test, y_train, y_valid, y_test = load_dataset(dataset, n_train=5000, d=d)
    kdt = neighbors.KDTree(x_train)
    _, index = kdt.query(x_test, k=k)
    predictions = np.sum(y_train[index], axis=1) / k
    return _RMSE(y_test, predictions)

def compare_performance(dataset='rosenbrock',d=1):
    """
    compare performance between knn using brute forced way and kd tree

    Inputs:
        dataset: (str) name of dataset
        d: (int) data dimensionality

    Outputs:
        [time taken with brute forced way,time taken with kd_tree]
    """
    k=5
    start = time.time()
    #brute force
    bf_loss = _test_regression(dataset=dataset,k=k,dist_metric='l2',d=2)
    bf_time = time.time()-start

    #kd_tree
    start = time.time()
    kd_loss = _kd_tree(dataset=dataset,k=k,dist_metric='l2',d=2)
    kd_time = time.time()-start


    return bf_time,kd_time

def run_Q3(d=[]):
    """
    script to compare performance between knn using brute forced way and kd tree on rosenbrock datasets with different dimensions

    Inputs:
        d: (int) data dimensionality

    Outputs:
        None
    """
    bf_time,kd_time = [],[]

    for i in d:
        b,k = compare_performance(dataset='rosenbrock',d=i)
        bf_time.append(b)
        kd_time.append(k)
    plot(xlabel='d',ylabel='time',name='compare_performance',x=d,y=[bf_time,kd_time],legend=['BruteForce','KDTree'])
    print(bf_time)
    print(kd_time)

def _svd_regression(dataset='mauna_loa'):
    """
    svd on regression dataset

    Inputs:
        dataset: (str) name of dataset

    Outputs:
        RMSE on predicted values
    """
    if dataset=='rosenbrock':
        x_train, x_valid, x_test, y_train, y_valid, y_test = load_dataset('rosenbrock', n_train=5000, d=2)
    else:
        x_train, x_valid, x_test, y_train, y_valid, y_test = load_dataset(dataset)

    x_total = np.vstack([x_train, x_valid])
    y_total = np.vstack([y_train, y_valid])

    X = np.ones((len(x_total), len(x_total[0]) + 1))
    X[:, 1:] = x_total

    U, S, Vh = np.linalg.svd(X)

    # Invert Sigma
    sig = np.diag(S)
    filler = np.zeros([len(x_total)-len(S), len(S)])
    sig_inv = np.linalg.pinv(np.vstack([sig, filler]))

    # Compute weights
    w = Vh.T @ (sig_inv @ (U.T @ y_total))

    # Make test predictions
    X_test = np.ones((len(x_test), len(x_test[0]) + 1))
    X_test[:, 1:] = x_test
    predictions = X_test @ w

    return _RMSE(y_test, predictions)

def _svd_classification(dataset='mnist_small'):
    """
    svd on classificaiton dataset

    Inputs:
        dataset: (str) name of dataset

    Outputs:
        accuracy on predicted values
    """
    if dataset=='rosenbrock':
        x_train, x_valid, x_test, y_train, y_valid, y_test = load_dataset('rosenbrock', n_train=5000, d=2)
    else:
        x_train, x_valid, x_test, y_train, y_valid, y_test = load_dataset(dataset)

    x_total = np.vstack([x_train, x_valid])
    y_total = np.vstack([y_train, y_valid])

    X = np.ones((len(x_total), len(x_total[0]) + 1))
    X[:, 1:] = x_total

    U, S, Vh = np.linalg.svd(X)

    # Invert Sigma
    sig = np.diag(S)
    filler = np.zeros([len(x_total)-len(S), len(S)])
    sig_inv = np.linalg.pinv(np.vstack([sig, filler]))

    # Compute weights
    w = Vh.T @ (sig_inv @ (U.T @ y_total))

    # Make test predictions
    X_test = np.ones((len(x_test), len(x_test[0]) + 1))
    X_test[:, 1:] = x_test
    predictions = np.argmax(X_test @ w, axis=1)
    y_test = np.argmax(1 * y_test, axis=1)

    return (predictions == y_test).sum() / len(y_test)

def run_Q4():
    """
    scirpt to run svd on regression and classificaiton dataset

    Inputs:
        None

    Outputs:
        loss on regression data,accuracy on classificaiton data
    """
    regression_data = ['mauna_loa','rosenbrock','pumadyn32nm']
    classificaiton_data = ['iris','mnist_small']
    loss,acc = [],[]
    for i in regression_data:
        loss.append(_svd_regression(dataset=i))
    for i in classificaiton_data:
        acc.append(_svd_classification(dataset=i))
    print(loss)
    print(acc)
    return loss,acc

if __name__ == '__main__':
    run_Q1(k_range=[1,31])
    run_Q2(k_range=[1,31])
    run_Q3(d=list(range(2,10)))
    run_Q4()
