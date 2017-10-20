#!/bin/bash
rm kmeans_noshows_base/*
mkdir kmeans_noshows_base

for (( s = 0; s < 10; sample++ )); do
	for (( i = 2; i < 10; i++ )); do
		
		Rscript clusters.R data/base_kmeans_sample_${s}.csv kmeans_noshows_base ${i}
		mv cls_plots.pdf kmeans_noshows_base/plots_cls${i}.pdf

	done	
done
