#!/bin/bash

# rm dic.txt
# echo "" > dic.txt

cd ap_noshows

cp ../csv_dealer.py .
rm *_clean.csv

for i in rules*.csv; 
do
	python csv_dealer.py $i
	mv fout.csv $(basename $i .csv)_clean.csv
done

rm csv_dealer.py

cd ../ap_noshows_freq_4

# rm ../dic.txt
# echo "" > ../dic.txt

cp ../csv_dealer.py .
rm *_clean.csv

for i in rules*.csv; 
do
	python csv_dealer.py $i
	mv fout.csv $(basename $i .csv)_clean.csv
done

rm csv_dealer.py

cd ..

