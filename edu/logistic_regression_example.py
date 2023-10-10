from import_libraries import *
spent = round(40 + 50*rand(100),0)
churn = 200 + 7*spent + 1000*(rand(100)-1/2)
churn_binary = (churn<700).astype(int)
churn_bin_transf = 0.98*churn_binary+0.01
Xtr = c_[ones(len(spent)), spent]
a = inv(Xtr.T@Xtr)@Xtr.T@logit(churn_bin_transf)
a
pred = 1/(1+exp(-Xtr@a))
plt.scatter(spent,churn_bin_transf)
plt.scatter(spent,pred)
plt.show()