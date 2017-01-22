import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


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

    return error_sqr


df_train = pd.read_table("crime-train.txt")
df_test = pd.read_table("crime-test.txt")
arr_train = df_train.values
arr_test = df_test.values

data_rows, data_columns = np.shape(arr_train)
data_rows_test, data_columns_test = np.shape(arr_test)
feas_rates = arr_train[:data_rows, 1:data_columns]
crime_rates = arr_train[:data_rows, :1]
feas_rates_test = arr_test[:data_rows_test, 1:data_columns_test]
crime_rates_test = arr_test[:data_rows_test, :1]
lambda_ = 600

age = df_train.columns.get_loc("agePct12t29")
pctWS = df_train.columns.get_loc("pctWSocSec")
pctKids = df_train.columns.get_loc("PctKids2Par")
pctIll = df_train.columns.get_loc("PctIlleg")
housVaca = df_train.columns.get_loc("HousVacant")
five_feas = [age, pctWS, pctKids, pctIll, housVaca]


while(lambda_ > 1):
    coeff = lasso_solver(np.log(lambda_), feas_rates, crime_rates)
    # 5.2
    #
    # shape = ["ro","y^","bs","gp","g*"]
    # for i in range (len(five_feas)):
    #     plt.plot(np.log(lambda_) ,coeff[five_feas[i]], shape[i])

    # 5.3
    # RSS = square_err(coeff, feas_rates, crime_rates)
    # plt.plot(np.log(lambda_), RSS, "ro")
    # lambda_ /= 2

    # 5.4
    RSS = square_err(coeff, feas_rates_test, crime_rates_test)
    plt.plot(np.log(lambda_), RSS, "ro")
    lambda_ /= 2

    # 5.5
    # coeff = lasso_solver(lambda_, feas_rates, crime_rates)
    # rows, columns = np.shape(np.nonzero(coeff))
    # plt.plot(lambda_, columns, "ro")
    # lambda_ /= 2

    # 5.7
    # coeff = lasso_solver(lambda_, feas_rates, crime_rates)
    # RSS = square_err(coeff, feas_rates_test, crime_rates_test)
    # plt.plot(lambda_, RSS, "ro")
    #lambda_ /= 2

# 5.7 (continue)
# The best performance lambda is approximately around 20
# best_lambda = 18
# best_coeff = lasso_solver(best_lambda, feas_rates_test, crime_rates_test)
# pos_highest_var_index = np.argmax(best_coeff)
# neg_lowest_var_index = np.argmin(best_coeff)
# print(pos_highest_var_index)
# print(neg_lowest_var_index)
# print(df_test.columns[pos_highest_var_index])
# print(df_test.columns[neg_lowest_var_index])

#plt.plot(,)
plt.show()
