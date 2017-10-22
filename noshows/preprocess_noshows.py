import pandas as pd
import datetime

dataset = pd.read_csv("data/noshows.csv")
# print len(dataset)
dataset_sample = dataset.sample(frac=0.1, replace=True)
# print len(dataset_sample)
# convert M e F para numeros
dataset['Gender'] = dataset['Gender'].replace('F',1)
dataset['Gender'] = dataset['Gender'].replace('M',0)

# print "Ja tratou dos sexos"
# drop da coluna sp porque estamos a trabalhar em clustering
dataset.drop('No-show',1,inplace=True)

di = set(dataset['Neighbourhood'])

dic = []
i=1
for d in di:
	dic.append((d, i))
	i+=1

dic= dict(dic)

for d in dic:
 	dataset['Neighbourhood'] = dataset['Neighbourhood'].replace(d,dic[d])

dataset.AppointmentDay = pd.to_datetime(pd.Series(dataset.AppointmentDay))
dataset.AppointmentDay = dataset.AppointmentDay.dt.day

dataset.ScheduledDay = pd.to_datetime(pd.Series(dataset.ScheduledDay))
dataset.ScheduledDay = dataset.ScheduledDay.dt.day


# print "Ja tratou das Localidades"

#  drop PatientId,AppointmentID

# ficheiro base com tudo em numeros
dataset.to_csv("data/base_noshows.csv", index=False)

# dataset.drop('PatientId',1,inplace=True)
dataset.drop('AppointmentID',1,inplace=True)

dataset.to_csv("data/base_kmeans_noshows.csv", index=False)

for i in range(10):
	sample = dataset.sample(frac=0.1,replace=False)
	filename = "data/base_kmeans_sample_" + str(i) + ".csv"
	sample.to_csv(filename, index=False)


# # Primeira tentativa: truncar os floats para int a bruta
# for key in dataset_sample.keys():
# 	dataset_sample[key] = dataset_sample[key].astype(int)

# dataset_sample.to_csv("data/truncint_noshows.csv",index=False)
