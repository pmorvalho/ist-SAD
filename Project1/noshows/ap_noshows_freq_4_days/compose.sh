rm -r composed
mkdir composed

for i in {5..100..15};
do 
 for k in {75..100..10};
 do 
   echo "=======================================" >> "case_s${i}.csv"
   echo "Rules with supp=${i} and conf=${k}" >> "case_s${i}.csv"
   echo "=======================================" >> "case_s${i}.csv"
   cat "rules_s${i}_c${k}_clean.csv" >> "case_s${i}.csv"
   echo "" >> "case_s${i}.csv"
   cat "interest_s${i}_c${k}.csv" >> "case_s${i}.csv" 
 done; 
done;

mv case*.csv composed/
