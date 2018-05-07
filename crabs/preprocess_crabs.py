import pandas as pd

dataset = pd.read_csv("data/crabs.csv")

# convert M e F para numeros
dataset['sex'] = dataset['sex'].replace('F',1)
dataset['sex'] = dataset['sex'].replace('M',0)

# drop da coluna sp porque estamos a trabalhar em clustering
dataset.drop('sp',1,inplace=True)

# ficheiro base com tudo em numeros
dataset.to_csv("data/base_crabs.csv", index=False)


# Primeira tentativa: truncar os floats para int a bruta
for key in dataset.keys():
	dataset[key] = dataset[key].astype(int)

dataset.to_csv("data/truncint_crabs.csv",index=False)