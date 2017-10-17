rm -r composed
mkdir composed

for i in {5..100..10};
do 
 for k in {50..100..10};
 do 
   cat "interest_s${i}_c${k}.csv" >> "case_s${i}_c${k}.csv"
   echo "" >> "case_s${i}_c${k}.csv"
   cat "rules_s${i}_c${k}.csv" >> "case_s${i}_c${k}.csv"
 done; 
done

mv case*.csv composed/
