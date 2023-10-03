import pickle
with open("list_of_clusters", "rb") as fp:
     clustered_multi_federated_data = pickle.load(fp)
series=clustered_multi_federated_data[0][0]
print(series)
