import sys
import os

def cvs_dealer(file, fout, file_name):
	for l in file:
		columns = l.split('"')
		prob = columns[1]
		prob = prob.split()
		has_prob = 0
		for d in dic:
			if d == prob[len(prob)-1]:
				# print "problem detected"
				has_prob = 1
		if has_prob==0:			
			fout.write(l)

dic = ['{Hipertension=0}', '{Scholarship=0}', '{Diabetes=0}', '{Alcoholism=0}', '{Handcap=0}']

if __name__ == "__main__":
    if len(sys.argv) != 2:
        err('USAGE: '+sys.argv[0]+' input.txt <model')    
    file_name = sys.argv[1]
    f = open(file_name)
    fout = open("fout.csv", "w+")
    cvs_dealer(f, fout, file_name)
    fout.close()
