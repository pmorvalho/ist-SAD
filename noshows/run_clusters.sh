#!/bin/bash
rm -r kmeans_noshows_base/*
mkdir kmeans_noshows_base

for (( s = 0; s < 10; s++ )); do
	for (( i = 2; i < 10; i++ )); do
		mkdir kmeans_noshows_base/sample_s${s}
		Rscript clusters.R data/base_kmeans_sample_${s}.csv kmeans_noshows_base/sample_s${s} ${i}
		mv cls_plots.pdf kmeans_noshows_base/sample_s${s}/plots_s${s}_cls${i}.pdf

	done	
done
