# install.packages("arules")
# install.packages("dplyr")
# install.packages("readr")
library(readr)
library(dplyr)
library(arules)
library(arulesViz)

discretize_all = function(table_d, type, n){
	for (i in 1:ncol(table_d)) {
		if (is.numeric(table_d[[i]])) {
			table_d[[i]]  =  discretize(table_d[[i]],  method  =  type,  categories  =  n, 
				ordered=TRUE)
		}
	}
	print(summary(table_d))
	return(table_d);
}

factor_all = function(table_d) {
	for (i in 1:ncol(table_d)) {
		table_d[[i]] = as.factor(table_d[[i]])
	}
	return(table_d)
}


apriori_all = function(dataset, folder) {
	for (s in seq(5, 100, by=15)) {
		for (c in seq(75, 100, by=10)) {
			s_i <- s / 100
			c_i <- c / 100
			print(c(s,"__",c))
			rules <- apriori(dataset, parameter=list(supp=s_i,conf=c_i,target="rules"))

			if (length(rules) != 0){
				plot(rules, measure=c("support", "confidence"), shading="lift")
				plot(rules, method="graph", control=list(type="itemsets"))

				subrules2 <- head(sort(rules, by="support"), 50)
				plot(subrules2, method="graph", control=list(type="itemsets"))			
			}

			# summary <- summary(rules)
			# write(summary, file=paste(c("ap_crabs/summary_s",s,"_c",c,".csv"), collapse=""))

			write(rules,
				file = paste(c(folder,"/","rules_s",s,"_c",c,".csv"), collapse=""),
	      		sep = ",", 
	      		quote = TRUE, 
	      		row.names = FALSE
	      	)

			interest <- interestMeasure(rules, c("support","confidence","lift","leverage","jaccard"), transactions=dataset)
			capture.output(interest, file=paste(c(folder,"/","interest_s",s,"_c",c,".csv"), collapse=""))
		}
	}

}

crabs_fact <- read_csv("data/truncint_crabs.csv")
crabs_fact = factor_all(crabs_fact)
#rules <- apriori(crabs_fact, parameter=list(supp=0.05,conf=0.5,target="rules"))
#plot(rules)

#discretize_all(dados, tipo de discretizacao, numero de bins)
crabs_d_fr <- read_csv("data/base_crabs.csv")
crabs_d_fr = discretize_all(crabs_d_fr, "frequency", 4)

crabs_d_int <- read_csv("data/base_crabs.csv")
crabs_d_int = discretize_all(crabs_d_int, "interval", 3)



# apriori_all(crabs_fact,"ap_crabs_trunc")

# apriori_all(crabs_d_fr,"ap_crabs_freq")

apriori_all(crabs_fact,"ap_crabs_trunc")

#plot(rules, measure=c("support", "confidence"), shading="lift"))
# plot(rules, method="graph", control=list(type="itemsets"))
# subrules2 <- head(sort(rules, by="support"), 20)
# plot(subrules2, method="graph", control=list(type="itemsets"))


# apriori(dados, parameter=list(support, confidence, target="rules"))
#rules <- apriori(crabs, parameter=list(supp=0.05,conf=0.5,target="rules"))



# neste caso, a funcao order ordena as regras pelo maior lift 
# e depois pela maior confidence
# ord_rules <- as(rules_i,"data.frame")
# ord_rules <- ord_rules[order(ord_rules$lift, ord_rules$confidence, decreasing=TRUE),]
# capture.output(ord_rules, file="cap_ord_rules.txt")

# discretize_all(dados, tipo de discretizacao, numero de bins)
# crabs <- read_csv("base_crabs.csv")
# crabs_d_fr = discretize_all(crabs, "frequency", 5)
# crabs_d_int = discretize_all(crabs, "interval", 5)

# rules_i <- apriori(crabs_d_int, parameter=list(supp=0.05,conf=0.5,target="rules"))
# inspect(rules_i)

# # devolve estatisticas resumidas sobre as regras geradas pelo apriori
# summary(rules_i)

# # escreve regras num ficheiro
# write(rules_i,
#       file = "rules_i.csv",
#       sep = ",",
#       quote = TRUE,
#       row.names = FALSE)

# # interestMeasure devolve metricas mais especializadas dos dados
# interest <- interestMeasure(rules_i, c("support","confidence","lift","leverage","jaccard"), transactions=crabs)
# capture.output(interest, file="interest_rules_i.txt")




