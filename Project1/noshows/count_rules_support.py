import sys
import os

def cvs_dealer(file, fout, file_name):
	dic=[[0.05,0],[0.1,0],[0.15,0],[0.2,0],[0.25,0]]
	dic=dict(dic)
	for l in file:
		columns = l.split('"')
		if len(columns) == 3 :
			prob = columns[2].split(",")
			val = float(prob[1]) 
			if val < 0.05 :
				dic[0.05] = int(dic[0.05]) + 1
			elif val < 0.1:
				dic[0.1] = int(dic[0.1]) + 1
			elif val < 0.15 :
				dic[0.15] = int(dic[0.15]) + 1	
			elif val < 0.2 :
				dic[0.2] = int(dic[0.2]) + 1
			elif val < 0.25 :
				dic[0.25] = int(dic[0.25]) + 1			
	return dic
		

if __name__ == "__main__":
    if len(sys.argv) != 2:
        err('USAGE: '+sys.argv[0]+' input.txt <model')    
    file_name = sys.argv[1]
    f = open(file_name)
    fout = open("rules_counter.csv", "w+")
    dic = cvs_dealer(f, fout, file_name)
    for d in dic:
    	fout.write(str(d) + "," + str(dic[d]) +"\n")
    fout.close