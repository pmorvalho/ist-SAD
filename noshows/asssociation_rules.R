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
			# write(summary, file=paste(c("ap_noshows/summary_s",s,"_c",c,".csv"), collapse=""))

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

noshows_base <- read_csv("data/base_noshows.csv") #_noAppointmentID.csv")
noshows_base = factor_all(noshows_base)
#rules <- apriori(noshows_base, parameter=list(supp=0.05,conf=0.5,target="rules"))
#plot(rules)

#discretize_all(dados, tipo de discretizacao, numero de bins)
noshows_d_fr <- read_csv("data/base_noshows.csv") #_noAppointmentID.csv")
noshows_d_fr = discretize_all(noshows_d_fr, "frequency", 4)

noshows_d_int <- read_csv("data/base_noshows.csv") #_noAppointmentID.csv")
noshows_d_int = discretize_all(noshows_d_int, "interval", 3)



apriori_all(noshows_base,"ap_noshows") #_noAppointmentID")

# apriori_all(noshows_d_fr,"ap_noshows_freq")

# apriori_all(noshows_fact,"ap_noshows_trunc")

#plot(rules, measure=c("support", "confidence"), shading="lift"))
# plot(rules, method="graph", control=list(type="itemsets"))
# subrules2 <- head(sort(rules, by="support"), 20)
# plot(subrules2, method="graph", control=list(type="itemsets"))


# apriori(dados, parameter=list(support, confidence, target="rules"))
#rules <- apriori(noshows, parameter=list(supp=0.05,conf=0.5,target="rules"))



# neste caso, a funcao order ordena as regras pelo maior lift 
# e depois pela maior confidence
# ord_rules <- as(rules_i,"data.frame")
# ord_rules <- ord_rules[order(ord_rules$lift, ord_rules$confidence, decreasing=TRUE),]
# capture.output(ord_rules, file="cap_ord_rules.txt")

# discretize_all(dados, tipo de discretizacao, numero de bins)
# noshows <- read_csv("base_noshows.csv")
# noshows_d_fr = discretize_all(noshows, "frequency", 5)
# noshows_d_int = discretize_all(noshows, "interval", 5)

# rules_i <- apriori(noshows_d_int, parameter=list(supp=0.05,conf=0.5,target="rules"))
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
# interest <- interestMeasure(rules_i, c("support","confidence","lift","leverage","jaccard"), transactions=noshows)
# capture.output(interest, file="interest_rules_i.txt")




