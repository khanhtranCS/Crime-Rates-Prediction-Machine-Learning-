import pandas as pd
import numpy as np

def lasso_solver(lambda_, X, vec_y):
    converge = 10
    a_term = np.dot(np.transpose(X), X)
    #print("hereadaf", a_term)
    #print("dfasd", np.linalg.inv(a_term + np.dot(lambda_,np.identity(len(a_term)))))
    new_w = np.dot(np.dot(np.linalg.inv(a_term + np.dot(lambda_,np.identity(len(a_term)))),np.transpose(X)), vec_y)
    prev_w = new_w
    #print(prev_w)
    x_rows, x_columns = np.shape(X)
    #print(X[0:1, :x_rows])
    #print("here", np.transpose(prev_w))
    #print(np.shape(X))
    while(converge >= 10**-6):
        prev_w = new_w
        for j in range(0, x_columns):
            a = 0
            c = 0
            for i in range(0, x_rows):
                a += X[i,j]**2
                #print(X[i:i+1, :x_rows])
                dot_pro = np.dot(X[i:i+1,:x_rows], prev_w)
                #dot_pro = np.dot(np.transpose(prev_w),X[i:i+1,:x_columns])
                #print(dot_pro)
                c += X[i,j]*(vec_y[i] - dot_pro + prev_w[j]*X[i,j])
            #print(2*c)

            new_w[j]= soft((2*c)/(2*a), lambda_/(2*a))
        converge = np.abs(np.linalg.norm(prev_w) - np.linalg.norm(new_w))
    return new_w


def soft(a, sigma):
    return (np.abs(a)/a) * np.maximum((np.abs(a) - sigma),0)

def square_err(coeff, feas_rates, actual_rate):
    model_rate = np.dot(feas_rates, coeff)
    result_y = 0
    error_vec = actual_rate - model_rate
    error_sqr = np.sum(np.square(error_vec))

    print(error_sqr)

def main():
    df_train = pd.read_table("crime-train.txt")
    df_test = pd.read_table("crime-test.txt")
    arr_train = df_train.values
    data_rows, data_columns = np.shape(arr_train)
    feas_rates = arr_train[:data_rows, 1:data_columns]
    crime_rates = arr_train[:data_rows, :1]
    #print(crime_rates)
    coeff = lasso_solver(600, feas_rates, crime_rates)
    #print(coeff)
    square_err(coeff, feas_rates, crime_rates)

main()