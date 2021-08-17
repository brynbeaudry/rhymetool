# Make map

python3 -m rhymetool map --input lyrics_to_sort.txt --level 4 --out map4.json --debug 1

# Merge maps together
# and sort them

python3 -m rhymetool merge_map --map1 map_syl_2_3.json --map2 map4.json --out map_syl_2_3_4.json --debug 1

# Make a list of rhymes from a rhyme map

python3 -m rhymetool make_list --level 3 --min-words 4 --lookback 4 --map-file map_syl_2_3_4.json --out output_2_3.txt --shuffle 0 --debug 1


# sort the rhymes in a file and output to another file. This will cut out lines in the middle that don't rhyme.
# It is best, at this stage to use an input file that was generated from make list (from map)
# Randomizing the lines would produce less results.
python3 -m rhymetool sort-file --level 3 --out output_sf1.txt --in output3_5.txt --debug 1


# Plans
Once you merge maps with end-end2-end3....
Then you basically have a sorted list of rhymes
Starrting with the more perfect rhymes, larger rhymes,

# Dump map.
# Strategy is just making a map, perhaps a big one, sorting appropriately, and then running this dump of the map with the rhymes sorts, randomizing the lists for the keys and preventing duplicate lines. Also leverages what we already know about lines

python3 -m rhymetool dump-map --level 3 --reduced-level 2 --out output_map_dump1.txt --input sorted_map1to5.json --debug 1 --shuffle 1

