#!/bin/bash

for dir in ap_noshows*;
do
	echo "$dir"
	cd $dir

	cp ../csv_dealer.py .
	rm *_clean.csv

	for i in rules*; 
	do
		python csv_dealer.py $i
		mv fout.csv $(basename $i .csv)_clean.csv
	done

	rm csv_dealer.py

	cd ..

done


