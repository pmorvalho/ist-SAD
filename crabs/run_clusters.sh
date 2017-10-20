#!/bin/bash
rm kmeans_crabs_base/*
mkdir kmeans_crabs_base


for (( i = 2; i < 10; i++ )); do
	Rscript clusters.R kmeans_crabs_base ${i}
	mv cls_plots.pdf kmeans_crabs_base/plots_cls${i}.pdf
done
