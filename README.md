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

# Good luck chump. Get off my lawn.

