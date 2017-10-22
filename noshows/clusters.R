library(readr)
library(dplyr)
library(ggplot2)
library(clv)

kmeans_wrapper = function(dataset, folder){
	noshows_fact <- read_csv(dataset)
	diss <- as.matrix(dist(noshows_fact))

	for (centroids in seq(2, 10, by=1)) {
		print(c("with ",centroids, " centroids..."))
		noshows_clust <- kmeans(noshows_fact, centroids)

		cls.scatt <- cls.scatt.data(noshows_fact, noshows_clust$cluster, dist="manhattan")

		dunn <- clv.Dunn(cls.scatt, intraclust, interclust)
		dunn <- capture.output(dunn)

		davies <- clv.Davies.Bouldin(cls.scatt, intraclust, interclust)
		davies <- capture.output(davies)

		# silhouette <- mean(silhouette(noshows_clust$cluster, dmatrix=diss^2)[,3])
		silhouette <- mean(silhouette(noshows_clust$cluster, dmatrix=diss)[,3])
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

		# pdf("cls_plots.pdf")
		# print(ggplot(noshows_fact, aes(AppointmentDay, SMS_received, color=noshows_clust$cluster)) + geom_point())
		# print(ggplot(noshows_fact, aes(AppointmentDay, ScheduledDay, color=noshows_clust$cluster)) + geom_point())
		# print(ggplot(noshows_fact, aes(ScheduledDay, SMS_received, color=noshows_clust$cluster)) + geom_point())
		# print(ggplot(noshows_fact, aes(Gender, Scholarship, color=noshows_clust$cluster)) + geom_point())
		# dev.off()
	}
}

args <- commandArgs(trailingOnly = TRUE)
intraclust = c("complete","average","centroid")
interclust = c("single", "complete", "average","centroid", "aveToCent", "hausdorff")

if (length(args) == 2) {
	kmeans_wrapper(args[1], args[2])	
}
