# install.packages("arules")
# install.packages("dplyr")
# install.packages("readr")
library(readr)
library(dplyr)
library(arules)

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
	for (s in seq(5, 100, by=45)) {
		for (c in seq(50, 100, by=25)) {
			s_i <- s / 100
			c_i <- c / 100
			print(c(s,"__",c))
			rules <- apriori(crabs_fact, parameter=list(supp=s_i,conf=c_i,target="rules"))

			# summary <- summary(rules)
			# write(summary, file=paste(c("ap_crabs/summary_s",s,"_c",c,".csv"), collapse=""))

			write(rules,
				file = paste(c(folder,"/","rules_s",s,"_c",c,".csv"), collapse=""),
	      		sep = ",", 
	      		quote = TRUE, 
	      		row.names = FALSE
	      	)

			interest <- interestMeasure(rules, c("support","confidence","lift","leverage","jaccard"), transactions=crabs_fact)
			capture.output(interest, file=paste(c(folder,"/","interest_s",s,"_c",c,".csv"), collapse=""))
		}
	}

}

crabs_fact <- read_csv("data/truncint_crabs.csv")
crabs_fact = factor_all(crabs_fact)

apriori_all(crabs_fact,"ap_crabs")


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




