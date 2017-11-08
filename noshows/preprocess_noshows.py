import pandas as pd
import datetime

dataset = pd.read_csv("data/noshows.csv")

# convert M e F para numeros
dataset['Gender'] = dataset['Gender'].replace('F',1)
dataset['Gender'] = dataset['Gender'].replace('M',0)

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

dataset.AppointmentDay = pd.to_datetime(pd.Series(dataset.AppointmentDay))
dataset.AppointmentDay = dataset.AppointmentDay.dt.week

dataset.ScheduledDay = pd.to_datetime(pd.Series(dataset.ScheduledDay))
dataset.ScheduledDay = dataset.ScheduledDay.dt.week


