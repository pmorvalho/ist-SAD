library(readr)
library(dplyr)
library(ggplot2)
library(clv)

kmeans_wrapper = function(dataset, folder, centroids){
	crabs_clust <- kmeans(dataset, centroids)
	cls.scatt <- cls.scatt.data(dataset, crabs_clust$cluster, dist="manhattan")

	dunn <- clv.Dunn(cls.scatt, intraclust, interclust)
	dunn <- capture.output(dunn)

	davies <- clv.Davies.Bouldin(cls.scatt, intraclust, interclust)
	davies <- capture.output(davies)

	silhouette <- mean(silhouette(crabs_clust$cluster, dmatrix=diss^2)[,3])
	silhouette <- capture.output(silhouette)

	# general stats file
	cat("\nDunn Index", file=paste(c(folder,"/","stats_cent",centroids,".csv"),collapse=""),sep="\n",append=TRUE)
	cat(dunn,file=paste(c(folder,"/","stats_cent",centroids,".csv"),collapse=""),sep="\n",append=TRUE)	
	cat("\nDavies Index", file=paste(c(folder,"/","stats_cent",centroids,".csv"),collapse=""),sep="\n",append=TRUE)
	cat(davies,file=paste(c(folder,"/","stats_cent",centroids,".csv"),collapse=""),sep="\n",append=TRUE)
	cat("\nSilhouette Coefficient", file=paste(c(folder,"/","stats_cent",centroids,".csv"),collapse=""),sep="\n",append=TRUE)
	cat(silhouette, file=paste(c(folder,"/","stats_cent",centroids,".csv"),collapse=""),sep="\n",append=TRUE)

	# only dunn file
	cat(c("\nWith ", centroids," centroids\n"),file=paste(c(folder,"/","all_dunn_indexes.csv"),collapse=""),sep="",append=TRUE)	
	cat(dunn,file=paste(c(folder,"/","all_dunn_indexes.csv"),collapse=""),sep="\n",append=TRUE)	


	# only davies files
	cat(c("\nWith ", centroids," centroids\n"),file=paste(c(folder,"/","all_davies_indexes.csv"),collapse=""),sep="",append=TRUE)	
	cat(davies,file=paste(c(folder,"/","all_davies_indexes.csv"),collapse=""),sep="\n",append=TRUE)	


	# only silhouette file
	cat(c("\nWith ", centroids," centroids\n"),file=paste(c(folder,"/","all_silhouettes.csv"),collapse=""),sep="",append=TRUE)	
	cat(silhouette, file=paste(c(folder,"/","all_silhouettes.csv"),collapse=""),sep="\n",append=TRUE)

	pdf("cls_plots.pdf")
	print(ggplot(dataset, aes(sex, FL, color=crabs_clust$cluster)) + geom_point())
	print(ggplot(dataset, aes(sex, RW, color=crabs_clust$cluster)) + geom_point())
	print(ggplot(dataset, aes(sex, CL, color=crabs_clust$cluster)) + geom_point())
	print(ggplot(dataset, aes(sex, CW, color=crabs_clust$cluster)) + geom_point())
	print(ggplot(dataset, aes(sex, BD, color=crabs_clust$cluster)) + geom_point())

	print(ggplot(dataset, aes(RW, FL, color=crabs_clust$cluster)) + geom_point())
	print(ggplot(dataset, aes(RW, CL, color=crabs_clust$cluster)) + geom_point())
	print(ggplot(dataset, aes(RW, CW, color=crabs_clust$cluster)) + geom_point())
	print(ggplot(dataset, aes(RW, BD, color=crabs_clust$cluster)) + geom_point())
	dev.off()
}

args <- commandArgs(trailingOnly = TRUE)
intraclust = c("complete","average","centroid")
interclust = c("single", "complete", "average","centroid", "aveToCent", "hausdorff")
crabs_fact <- read_csv("data/base_crabs.csv")
diss <- as.matrix(dist(crabs_fact))

if (length(args) == 2) {
	kmeans_wrapper(crabs_fact, args[1], args[2])	
}
