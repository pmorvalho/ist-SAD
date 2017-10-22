import pandas as pd
import datetime

dataset = pd.read_csv("data/noshows.csv")

# convert M e F para numeros
dataset['Gender'] = dataset['Gender'].replace('F',1)
dataset['Gender'] = dataset['Gender'].replace('M',0)

# drop da coluna sp porque estamos a trabalhar em clustering
dataset.drop('No-show',1,inplace=True)

# di fica com as varias localidades unicas do dataset
di = set(dataset['Neighbourhood'])

dic = []
i=1
# para cada uma das localidades (d) vamos atribuir um valor numerico (i) e colocar numa lista (dic)
for d in di:
	dic.append((d, i))
	i+=1

#fazemos da lista dic um dicionario
dic= dict(dic)

#alteramos cada uma das localicadas pelo respectivo valor numerico do dicionario
for d in dic:
 	dataset['Neighbourhood'] = dataset['Neighbourhood'].replace(d,dic[d])

#para cada data (d) pomos na lista dic um tuplo da data com o respectivo numero da semana do ano da data
dic = []
for d in dataset['ScheduledDay']:
	date = datetime.datetime.strptime( d, "%Y-%m-%dT%H:%M:%SZ" )
	dic.append((d, date.timetuple().tm_yday))

# fazemos um dicionario da lista e fazemos uma alteracao no dataset cada data passa a ser o numero da semana respectiva nesse ano
dic = dict(dic)
for d in dic:
	dataset['ScheduledDay'] = dataset['ScheduledDay'].replace(d,dic[d])

#repetimos o que fizemos para o ScheduleDay para o AppointmentDay
dic = []
for d in dataset['AppointmentDay']:
	date = datetime.datetime.strptime( d, "%Y-%m-%dT%H:%M:%SZ" )
	dic.append((d, date.timetuple().tm_yday))

dic = dict(dic)
for d in dic:
	dataset['AppointmentDay'] = dataset['AppointmentDay'].replace(d,dic[d])

# ficheiro base com tudo em numeros
dataset.to_csv("data/base_noshows_days.csv", index=False)
