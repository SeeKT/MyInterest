import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import statsmodels.api as sm
from sklearn.datasets import load_boston

######### Regression using statsmodels.api #########
def execute_regression(exp_variable, obj_variable):
    # add constants
    X = sm.add_constant(exp_variable)
    model = sm.OLS(obj_variable, X)
    # execute regression
    results = model.fit()
    # output results
    return results
####################################################

# read data from sample
boston = load_boston()
df = pd.DataFrame(boston.data, columns=boston.feature_names)
obj_y = boston.target
expression_label = df.columns.values
data_label = ['price']; data_list = [obj_y]
for i in range(len(expression_label)):
    data_label.append(expression_label[i])
    data_list.append(df.iloc[:, i].values)
data_dict = dict(zip(data_label, data_list))
new_df = pd.DataFrame(data_dict)

# objective variable and expression variable
y = new_df.iloc[:, 0]; x = new_df.iloc[:, 6]
# correlation matrix
correlation_matrix = np.corrcoef(x.values, y.values)
# execute regression analysis
regression_result = execute_regression(exp_variable=x, obj_variable=y)
print(regression_result.summary())
#print(regression_result.tvalues)
b0 = regression_result.params[0]; b1 = regression_result.params[1]
# figure
"""
fig = plt.figure()
ax = fig.add_subplot(111)
ax.scatter(x.values, y.values, c = 'k', s = 5, label = 'corr = %.2f' % (correlation_matrix[0][1]))
ax.plot(x.values, b0 + b1 * x.values, c = 'r', label = r'$y = %.2f + %.2fx$' % (b0, b1))
ax.set_xlabel(new_df.columns.values[6])
ax.set_ylabel(new_df.columns.values[0])
ax.grid(True); ax.legend(loc = 'best')
plt.show()
"""