#!/bin/bash
rm kmeans_crabs_base_not_pca/*
mkdir kmeans_crabs_base_not_pca


for (( i = 2; i < 20; i++ )); do
	Rscript clusters.R kmeans_crabs_base_not_pca ${i}
	mv cls_plots.pdf kmeans_crabs_base_not_pca/plots_cls${i}.pdf
done
