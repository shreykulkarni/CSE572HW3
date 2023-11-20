import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from surprise import Dataset, Reader, accuracy, KNNWithMeans
from surprise.accuracy import rmse, mae
from surprise.model_selection import cross_validate, train_test_split
from surprise.prediction_algorithms.matrix_factorization import SVD
from surprise.prediction_algorithms.knns import KNNBasic
from surprise.prediction_algorithms.matrix_factorization import NMF

data = pd.read_csv('ratings_small.csv')
#3a
reader = Reader(line_format='user item rating timestamp', sep=',', skip_lines=1)
dataset = Dataset.load_from_df(data[['userId', 'movieId', 'rating']], reader)


#PMF
pmf = SVD(biased=False)
#User-based Collaborative Filtering
user_cf = KNNBasic(sim_options={'user_based': True})
#Item-based Collaborative Filtering
item_cf = KNNBasic(sim_options={'user_based': False})  

#Function to compute averages
def computeAverages(model):
    cross_validated_metrics = cross_validate(model, dataset, measures=['MAE', 'RMSE'], cv=5, verbose=True)
    avg_mae = sum(cross_validated_metrics['test_mae']) / 5
    avg_rmse = sum(cross_validated_metrics['test_rmse']) / 5
    return avg_mae, avg_rmse


pmf_mae, pmf_rmse = computeAverages(pmf)
user_cf_mae, user_cf_rmse = computeAverages(user_cf)
item_cf_mae, item_cf_rmse = computeAverages(item_cf)

#3c
print(f"Average MAE of Probabilistic Matrix Factorization: {pmf_mae}")
print(f"Average RMSE of Probabilistic Matrix Factorization: {pmf_rmse}")
print(f"Average MAE of User-based Collaborative Filtering: {user_cf_mae}")
print(f"Average RMSE of User-based Collaborative Filtering: {user_cf_rmse}")
print(f"Average MAE of Item-based Collaborative Filtering: {item_cf_mae}")
print(f"Average RMSE of Item-based Collaborative Filtering: {item_cf_rmse}")


#3e
userBasedCosine = KNNBasic(sim_options={'name': 'cosine', 'user_based': True})
userBasedCosineMAE, userBasedCosineRMSE = computeAverages(userBasedCosine)
userBasedMSD = KNNBasic(sim_options={'name': 'msd', 'user_based': True})
userBasedMSDMAE, userBasedMSDRMSE = computeAverages(userBasedMSD)
userBasedPearson = KNNBasic(sim_options={'name': 'pearson', 'user_based': True})
userBasedPearsonMAE, userBasedPearsonRMSE = computeAverages(userBasedPearson)

itemBasedCosine = KNNBasic(sim_options={'name': 'cosine', 'user_based': False})
itemBasedCosineMAE, itemBasedCosineRMSE = computeAverages(itemBasedCosine)
itemBasedMSD = KNNBasic(sim_options={'name': 'msd', 'user_based': False})
itemBasedMSDMAE, itemBasedMSDRMSE = computeAverages(itemBasedMSD)
itemBasedPearson = KNNBasic(sim_options={'name': 'pearson', 'user_based': False})
itemBasedPearsonMAE, itemBasedPearsonRMSE = computeAverages(itemBasedPearson)


plt.figure(figsize=(12, 8))
plt.plot(['Cosine', 'MSD', 'Pearson'],
[userBasedCosineMAE, userBasedMSDMAE, userBasedPearsonMAE], label='MAE')
plt.plot(['Cosine', 'MSD', 'Pearson'],
[userBasedCosineRMSE, userBasedMSDRMSE, userBasedPearsonRMSE], label='RMSE')
plt.legend()
plt.title('Cosine, MSD, Pearson similarities Impact on User-based CF Performance')
plt.ylabel('Error')
plt.show()

plt.figure(figsize=(12, 8))
plt.plot(['Cosine', 'MSD', 'Pearson'],
[itemBasedCosineMAE, itemBasedMSDMAE, itemBasedPearsonMAE], label='MAE')
plt.plot(['Cosine', 'MSD', 'Pearson'],
[itemBasedCosineRMSE, itemBasedMSDRMSE, itemBasedPearsonRMSE], label='RMSE')
plt.legend()
plt.title('Cosine, MSD, Pearson similarities Impact on Item-based Collaborative Filtering Performance')
plt.ylabel('Error')
plt.show()


#3f
userBasedMAEValues = []
userBasedRMSEValues = []
itemBasedMAEValues = []
itemBasedRMSEValues = []
kValues = range(1, 20)

for i in kValues:
    print(i)
    userBased = KNNBasic(k=i, sim_options={'user_based': True})
    userBasedCompute = cross_validate(userBased, dataset, measures=['MAE', 'RMSE'], cv=5)
    userBasedMAE = userBasedCompute['test_mae'].mean()
    userBasedMAEValues.append(userBasedMAE)
    userBasedRMSE = userBasedCompute['test_rmse'].mean()
    userBasedRMSEValues.append(userBasedRMSE)
    itemBased = KNNBasic(k=i, sim_options={'user_based': False}) 
    itemBasedCompute = cross_validate(itemBased, dataset, measures=['MAE', 'RMSE'], cv=5)
    itemBasedMAE = itemBasedCompute['test_mae'].mean()
    itemBasedMAEValues.append(itemBasedMAE)
    itemBasedRMSE = itemBasedCompute['test_rmse'].mean()
    itemBasedRMSEValues.append(itemBasedRMSE)
    
    
plt.plot(kValues, userBasedMAEValues, label='User-based Collaborative Filtering')
plt.plot(kValues, itemBasedMAEValues, label='Item-based Collaborative Filtering')
plt.title('Relationship Between Number of Neighbors and Performance on MAE')
plt.xlabel('K')
plt.ylabel('MAE')
plt.legend()
plt.show()

plt.plot(kValues, userBasedRMSEValues, label='User Collaborative Filtering')
plt.plot(kValues, itemBasedRMSEValues, label='Item Collaborative Filtering')
plt.title('Relationship Between Number of Neighbors and Performance on RMSE')
plt.xlabel('K')
plt.ylabel('RMSE')
plt.legend()
plt.show()


#3g
userK = min(user_rmses)
itemK = min(item_rmses)

userBestK = user_rmses.index(userK) + 1
itemBestK  =item_rmses.index(itemK) + 1

print(f"The best value of K for User-based CF: {userBestK}, Where RMSE is: {userK}")
print(f"The best value of K for Item-based CF: {itemBestK}, Where RMSE is: {itemK}")
