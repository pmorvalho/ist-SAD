#!/bin/bash
rm -r kmeans_noshows_base/*
mkdir kmeans_noshows_base

#rm -r kmeans_noshows_days/*
#mkdir kmeans_noshows_days

for (( s = 0; s < 10; s++ )); 
do
	echo ${s}
	mkdir kmeans_noshows_base/sample_s${s}
	Rscript clusters.R data/base_kmeans_sample_${s}.csv kmeans_noshows_base/sample_s${s}

	#mkdir kmeans_noshows_days/sample_s${s}
	#Rscript clusters.R data/base_kmeans_sample_${s}.csv kmeans_noshows_days/sample_s${s}
done
