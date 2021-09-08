# Make map

python3 -m rhymetool map --input "$INPUT_LINES.txt" --level 5 --out "./maps/$current_time/$i.json" --debug 0

# Merge maps together and sort them

python3 -m rhymetool merge_map --map1 ./maps/$current_time/merged_map.json --map2 "./maps/$current_time/$i.json" --out ./maps/$current_time/merged_map.json --debug 0

# Make a list of rhymes from a rhyme map. (Depriciated)

python3 -m rhymetool make_list --level 3 --min-words 4 --lookback 4 --map-file map_syl_2_3_4.json --out output_2_3.txt --shuffle 0 --debug 1

# Dump map

python3 -m rhymetool dump-map --level $i --reduced-level 0 --out ./maps/$current_time/dump.txt --mode a --input "./maps/$current_time/merged_map.json" --debug 0 --shuffle 1

# Delete duplicate lines without sorting
awk '!a[$0]++' lyr1.txt > lyr2.txt  


# An example implmentation
# Make maps at various levels
for ((i=$HIGH; i>=$LOW; i--))
do
    python3 -m rhymetool map --input "./input/$INPUT_LINES.txt" --level $i --out "./maps/$current_time/$i.json" --debug 0
    echo "Made map $i"
done

# Merge the results together
echo '{}' > ./maps/$current_time/merged_map.json
for ((i=$HIGH; i>=$LOW; i--))
do
    python3 -m rhymetool merge_map --map1 ./maps/$current_time/merged_map.json --map2 "./maps/$current_time/$i.json" --out ./maps/$current_time/merged_map.json --debug 0
    echo "Merged map $i"
done
# Dump the map at various levels into one file
for ((i=$HIGH; i>=$LOW; i--))
do
    python3 -m rhymetool dump_map_var --level $i --reduced-level 0 --out ./maps/$current_time/var_dump.txt --mode a --input "./maps/$current_time/merged_map.json" --debug 0 --shuffle 1
    echo "Dumped merged map with level $i, with some shuffling and rules"
done
echo "Done"

# Straight up dump the map with no rules, except shuffling by default, cause why not
python3 -m rhymetool dump_map --out ./maps/$current_time/dump.txt --input "./maps/$current_time/merged_map.json"
echo "Dumped merged map with no rules"
echo "Done"
awk '!a[$0]++' ./maps/$current_time/dump.txt > ./maps/$current_time/dump_unique.txt 


# Delete duplicate lines without sorting
awk '!a[$0]++' ./maps/$current_time/var_dump.txt > ./maps/$current_time/var_unique.txt

# Produce cleaned versions of the the varied, unique rhyme dump starting strong with the cleaned level, and loosening to 2
for ((i=$CLEAN_LEVEL; i>=2; i--))
do
    python3 -m rhymetool clean_list --level $i --input ./maps/$current_time/dump_unique.txt --out "./maps/$current_time/cleaned_var$i.txt" --mode w
    echo "Cleaned varied with level $i"
done
echo "Done"

# Good luck chump. Get off my lawn.

