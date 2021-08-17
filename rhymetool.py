"""Predict from a previously generated song model."""
import argparse
import json
from os import error
from collections import OrderedDict
import math
import datetime
import sys
import random
import re
from timeit import default_timer as timer
from datetime import timedelta

import nltk
nltk.download('cmudict')

entries = nltk.corpus.cmudict.entries()

def rhyme(inp, level):
     syllables = [(word, syl) for word, syl in entries if word == inp]
     rhymes = []
     for (word, syllable) in syllables:
             rhymes += [word for word, pron in entries if pron[-level:] == syllable[-level:]]
     return set(rhymes)

def isNearRhyme(key1, word2, level=2):
    # write a function for use only with your maps
    # you can determine 
    # key will be the backward pronounciation
    # so if you take the reverse of the key, and split it, you can check all the levels based on some criteria.
    word1_syllables_rev = key1.split('_')
    word2_syllables = [syl for word, syl in entries if word == word][0]
    matches = len([x for x in word1_syllables_rev if x in word2_syllables])
    return matches >= level

def isRhymeSequentialWithKey(key1, word2, level=2):
    # Given the key that the word is from,
    # here the level is how many syllables have to match from left to right
    #start = timer()
    matches = 0
    word1_syllables_rev = key1.split('_')
    word2_syllables = [syl for word, syl in entries if word == word2][0]
    word2_syllables_rev = [x for x in word2_syllables[::-1]]
    min_length = min(len(word1_syllables_rev), len(word2_syllables_rev))
    for i in range(0,min_length):
        if word1_syllables_rev[i] == word2_syllables_rev[i]:
            matches = matches + 1
    #end = timer()
    #print(f'isRhyme for {word2} : {word1_syllables_rev} vs {word2_syllables_rev} :\n level: {level} matches: {matches} took : {timedelta(seconds=end-start)}')
    return matches >= level

def doTheyRhyme(word1, word2, level=1):
    # first, we don't want to report 'glue' and 'unglue' as rhyming words
    # those kind of rhymes are LAME
    if word1.find(word2) == len(word1) - len(word2):
        return False
    if word2.find(word1) == len(word2) - len(word1): 
        return False
    return word1 in rhyme(word2, level)

def isContainSameWord(word1, word2):
    # first, we don't want to report 'glue' and 'unglue' as rhyming words
    # those kind of rhymes are LAME
    if word1.find(word2) == len(word1) - len(word2):
        return True
    if word2.find(word1) == len(word2) - len(word1): 
        return True


def make_map(args):
    # Produces a rhyme map from a list of lines
    # Probably a good idea to edit list down to lines that make sense, and that you actually might use.
    # Based on the level provided
    rhyme_map = dict()
    filename = args.input
    level = args.level
    map_file = args.out
    file = open(filename,'r')
    text_lines = file.readlines()
    for idx, line in enumerate(text_lines):
        if args.debug:
            print(idx)
            if idx > 900:
                import pdb; pdb.set_trace();
        last_word = line.strip().split(' ')[-1]
        try:
            syllable_arr = [ syllables for word, syllables in entries if word == last_word][0]
            rhyming_syls = [ x for x in syllable_arr[-level:]]
            # Take the reverse
            rhyming_syls_rev = rhyming_syls[::-1]
            rhyme_key = '_'.join(rhyming_syls_rev)
            rhyme_map.setdefault(rhyme_key, []).append(line.strip())
        except IndexError:
            continue
        except:
            print(f'Line: {line}\nLast Word {last_word}\nSyllables used:{rhyme_key}\n')
            raise
    file.close()

    with open(map_file, 'w') as convert_file:
        convert_file.write(json.dumps(rhyme_map))
    print('Finished file mapping')

def dump_map(args):
    # Dump a map which has its rhymes in order.
    is_shuffle = args.shuffle
    reduced_level = args.reduced_level
    minimum_words = args.min_words
    with open(args.input, 'r') as f:
        sorted_rhyme_map = json.load(f)
    output = open(args.out, args.mode)
    results = []
    used_words = []
    for key, lines in sorted_rhyme_map.items():
        if is_shuffle:
            slines = list(set(lines.copy()))
            random.shuffle(slines)
            lines = slines
        for idx, line in enumerate(lines):
            this_word = line.split(' ')[-1].strip()
            is_enough_words = not (len(line.split(' ')) < minimum_words)
            if idx > 0:
                this_word = line.split(' ')[-1].strip()            
                previous_line = lines[idx-1]
                previous_word = previous_line.split(' ')[-1].strip()
                if not is_enough_words or this_word == previous_word or isContainSameWord(this_word, previous_word):
                    continue
                elif this_word in used_words:
                    continue
                elif (isRhymeSequentialWithKey(key, previous_word, args.level) or \
                    ( reduced_level and isRhymeSequentialWithKey(key, previous_word, reduced_level) ) ):
                    print(' . ', end=' ')
                    if previous_word not in used_words:
                        print(' . ', end=' ')
                        if args.debug:
                            sys.stdout.write(f'{previous_line}\n')
                        results.append(previous_line)
                        used_words.append(previous_word)
                    if args.debug:
                        sys.stdout.write(f'{line}\n')
                    results.append(line)
                    used_words.append(this_word)
            else:
                if len(used_words) \
                    and is_enough_words \
                    and this_word not in used_words \
                    and len(used_words) \
                    and (doTheyRhyme(this_word, used_words[-1], args.level) or \
                    ( reduced_level and doTheyRhyme(this_word, used_words[-1], reduced_level) ) ):
                    results.append(line)
                    used_words.append(this_word)
                    if args.debug:
                        sys.stdout.write(f'{line}\n')
                elif is_enough_words and this_word not in used_words:
                    results.append(line)
                    used_words.append(this_word)
                    if args.debug:
                        sys.stdout.write(f'{line}\n')

    for r in results:
        output.write(f'{r}\n')
    output.close()
    sys.stdout.flush()


def sort_file(args):
    reduced_level = args.reduced_level
    file_lines = []
    with open(args.input, 'r') as f:
        file_lines = f.read().splitlines()
    output = open(args.out, args.mode)
    results = []
    used_words = []
    for idx, line in enumerate(file_lines):
        this_word = line.split(' ')[-1].strip()
        if idx > 1:
            this_word = line.split(' ')[-1].strip()            
            previous_line = file_lines[idx-1]
            previous_word = previous_line.split(' ')[-1].strip()
            is_rhyme_with_previous = (doTheyRhyme(this_word, previous_word, args.level) or \
                ( reduced_level and doTheyRhyme(this_word, previous_word, reduced_level) ) )
            if this_word == previous_word or isContainSameWord(this_word, previous_word):
                continue
            elif this_word in used_words:
                continue
            elif is_rhyme_with_previous:
                print(' . ', end=' ')
                if previous_word not in used_words:
                    print(' . ', end=' ')
                    if args.debug:
                        sys.stdout.write(f'{previous_line}\n')
                    results.append(previous_line)
                    used_words.append(previous_word)
                if args.debug:
                    sys.stdout.write(f'{line}\n')
                results.append(line)
                used_words.append(this_word)
        else:
            results.append(line)
            used_words.append(this_word)
    for r in results:
        output.write(f'{r}\n')
    output.close()
    sys.stdout.flush()


def make_list(args):
    # opens a rhyme map and creates a preliminary list of rhyming entries
    # based on a lookback param, which can allow for alot more leniency for the rhymes
    rejected_lines = []
    minimum_words = args.min_words
    is_shuffle = args.shuffle
    with open(args.map_file) as f:
        rhyme_map = json.load(f)
    print('Loaded rhyme map')

    output = open(args.out, args.mode)

    used_lines = []
    used_words = []
    lookback = args.lookback
    reduced_level = args.reduced_level
    for key, lines in rhyme_map.items():
        #import pdb; pdb.set_trace();
        if is_shuffle:
            slines = list(set(lines.copy()))
            random.shuffle(slines)
            lines = slines
        for idx, line in enumerate(lines):
            if args.debug:
                sys.stdout.write(' . ')
            is_enough_words = not (len(line.split(' ')) < minimum_words)
            curr_line_last_word = line.split(' ')[-1].strip()
            if idx > 0:
                previous_line = lines[idx -1]
                prev_line_last_word = previous_line.split(' ')[-1].strip()
                this_and_last_contain_same_word = isContainSameWord(curr_line_last_word, prev_line_last_word)
                #import pdb; pdb.set_trace();
                if not is_enough_words or curr_line_last_word == prev_line_last_word or curr_line_last_word in used_words or this_and_last_contain_same_word:
                    rejected_lines.append(line)
                    if args.debug:
                        sys.stdout.write(' . ')
                    continue
                if doTheyRhyme(curr_line_last_word, prev_line_last_word, args.level) or \
                    ( reduced_level and doTheyRhyme(curr_line_last_word, prev_line_last_word, reduced_level ) ):
                    print(' . ', end=' ')
                    if prev_line_last_word not in used_words:
                        if args.debug:
                            sys.stdout.write(f'{previous_line}\n')
                        output.write(f'{previous_line}\n')
                        used_words.append(prev_line_last_word)
                    if args.debug:
                        sys.stdout.write(f'{line}\n')
                    output.write(f'{line}\n')
                    used_words.append(curr_line_last_word)
                elif lookback and len(used_lines) > lookback:
                    lookback_lines = used_lines[-lookback:]
                    for ul in lookback_lines:
                        used_line_last_word = ul.split(' ')[-1].strip()
                        if curr_line_last_word != used_line_last_word and \
                            doTheyRhyme(curr_line_last_word, used_line_last_word, args.level) or \
                            ( reduced_level and doTheyRhyme(curr_line_last_word, used_line_last_word, reduced_level ) ):
                            if args.debug:
                                sys.stdout.write(f'{line}\n')
                            output.write(f'{line}\n')
                            used_lines.append(line)
                            used_words.append(curr_line_last_word)
                            # Only write the line for one of the lookback lines
                            break
            else:
                if is_enough_words:
                    if args.debug:
                        sys.stdout.write(f'{line}\n')
                    output.write(f'{line}\n')
                    used_lines.append(line)
                    used_words.append(curr_line_last_word)
        if args.include_rejected:
            # Prints the rejected lines 
            if args.debug:
                output.write('=================================================================\n')
                sys.stdout.write('=================================================================\n')
                for rl in rejected_lines:
                    # now just print all the rejected lines for this rhyme in order.
                    # theoretically, you could randomize the input text to get different results.
                    output.write(f'{rl}\n')
                    if args.debug:
                        sys.stdout.write(f'{rl}\n')
        rejected_lines = []
    output.close()
    sys.stdout.flush()


def merge_map(args):
    # Creates a merged map, can potentially merge maps of different syllables
    # Ensures there is not repetition of the lines lists associated with syllable endings.
    merged_map = dict()
    map1 = open(args.map1, 'r')
    rhyme_map1 = json.load(map1)
    map1.close()
    map2 = open(args.map2, 'r')
    rhyme_map2 = json.load(map2)
    map2.close()

    for k, v in rhyme_map1.items():
        merged_map[k] = set(v)
    print('loaded map')
    for k, v in rhyme_map2.items():
        if merged_map.get(k) != None:
            merged_map[k] = merged_map[k] | set(v)
        else:
            merged_map[k] = set(v)
    print('Merged second map')


    # sort map by keys
    merged_map = OrderedDict(sorted(merged_map.items(),key=lambda item: (-len(item), item)))

    merged_map = { k: list(v) for k,v in merged_map.items() }

    merged_map = OrderedDict(reversed(list(merged_map.items())))

    print('Turned map values from sets to lists')
    with open(args.out, 'w') as output:
        output.write(json.dumps(merged_map))
    print('Done!')



def cli():
    # Todo, you ccould pick a certain number of a key before moving on, and then delete those lines from the list in that key, and
    # Then continue, until all the lines are used. This allows you to pepper in rhymes of different types.
    # Todo instead of print to console and out, append to results and then do a custom shuffile of the list
    # Todo take advantage of multithreading https://stackoverflow.com/questions/2846653/how-can-i-use-threading-in-python for mapping
    # Todo: Add a reverse option to map_dunp
    # Rhyme sequential is a much looser rhyme function, you can try to do a fall through, straight up rhyme, sequential rhyme, then near rhyme
    # Could also tighten up by specifying the desired rhyme length, and thus, the near rhyme checker would would better
    timestamp = str(datetime.datetime.now()).replace(' ','_')
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers()
    make_map_parser = subparsers.add_parser(
        "map", help="calls the make map function"
    )
    make_map_parser.add_argument(
        "--input",
        default='./lyrics_to_sort.txt',
        type=str,
        help="File with lines of lyrics",
    )
    make_map_parser.add_argument(
        "--level",
        default=1,
        type=int,
        help="Increase the level to increase the criteria for how good the rhyme should be. If one, the last syl pronounce needs to match, if two, the last two need to match",
    )

    make_map_parser.add_argument(
        "--out",
        default=f'map_syl_{timestamp}.json',
        type=str,
        help="The output file for stored a created map",
    )

    make_map_parser.add_argument(
        "--debug",
        default=0,
        type=int,
        help="debug val",
    )

    make_map_parser.set_defaults(func=make_map)

    make_list_parser = subparsers.add_parser(
        "make_list", help="calls the make list from map function"
    )

    make_list_parser.add_argument(
        "--level",
        default=2,
        type=int,
        help="Increase the level to increase the criteria for how good the rhyme should be. If one, the last syl pronounce needs to match, if two, the last two need to match",
    )

    make_list_parser.add_argument(
        "--min-words",
        default=4,
        type=int,
        help="Minimum words a line needs to have to be printed",
    )

    make_list_parser.add_argument(
        "--lookback",
        default=1,
        type=int,
        help="How far to look back in the list for rhymes.",
    )

    make_list_parser.add_argument(
        "--map-file",
        default=f'map_syl_{timestamp}.json',
        type=str,
        help="The created map precreated map file",
    )

    make_list_parser.add_argument(
        "--out",
        default=f'output{timestamp}.text',
        type=str,
        help="The output file for resulting lyrics",
    )

    make_list_parser.add_argument(
        "--mode",
        default='w',
        type=str,
        help="Write mode, defaults to w, but could be 'a'. Don't make it r",
    )

    make_list_parser.add_argument(
        "--include-rejected",
        default=False,
        type=bool,
        help="The output file for stored a created map",
    )

    make_list_parser.add_argument(
        "--shuffle",
        default=0,
        type=int,
        help="Whether or not to randomize the order of the dictionary, and the lookback lines if lookback lines are specified",
    )

    make_list_parser.add_argument(
        "--reduced-level",
        default=0,
        type=int,
        help="The reduced level to check if a rhyme can't be found for the original level",
    )

    make_list_parser.add_argument(
        "--debug",
        default=0,
        type=int,
        help="debug val",
    )

    make_list_parser.set_defaults(func=make_list)

    merge_map_parser = subparsers.add_parser(
        "merge_map", help="merges two maps"
    )

    merge_map_parser.add_argument(
        "--map1",
        default=f'map1.json',
        type=str,
        help="One map to merge",
    )

    merge_map_parser.add_argument(
        "--map2",
        default=f'map2.json',
        type=str,
        help="Another map to merge",
    )

    merge_map_parser.add_argument(
        "--out",
        default=f'merged_map{timestamp}.json',
        type=str,
        help="The output file for resulting map",
    )

    merge_map_parser.add_argument(
        "--debug",
        default=0,
        type=int,
        help="debug val",
    )

    merge_map_parser.set_defaults(func=merge_map)

    sort_file_parser = subparsers.add_parser(
        "sort-file", help="calls the sort file function"
    )

    sort_file_parser.add_argument(
        "--level",
        default=1,
        type=int,
        help="Increase the level to increase the criteria for how good the rhyme should be. If one, the last syl pronounce needs to match, if two, the last two need to match",
    )

    sort_file_parser.add_argument(
        "--input",
        default=f'list_in.txt',
        type=str,
        help="The output file for resulting map",
    )

    sort_file_parser.add_argument(
        "--out",
        default=f'list_out.txt',
        type=str,
        help="The output file for resulting map",
    )

    sort_file_parser.add_argument(
        "--reduced-level",
        default=0,
        type=int,
        help="The reduced level to check if a rhyme can't be found for the original level",
    )

    sort_file_parser.add_argument(
        "--max-tries",
        default=4,
        type=int,
        help="The max tries to get a rhyme from that particular pronounciation",
    )

    sort_file_parser.add_argument(
        "--mode",
        default='w',
        type=str,
        help="Write mode, defaults to w, but could be 'a'. Don't make it r",
    )

    sort_file_parser.add_argument(
        "--debug",
        default=0,
        type=int,
        help="debug val",
    )

    sort_file_parser.set_defaults(func=sort_file)

    dump_map_parser = subparsers.add_parser(
        "dump-map", help="calls the dump map function"
    )

    dump_map_parser.add_argument(
        "--level",
        default=1,
        type=int,
        help="Increase the level to increase the criteria for how good the rhyme should be. If one, the last syl pronounce needs to match, if two, the last two need to match",
    )

    dump_map_parser.add_argument(
        "--input",
        default=f'list_in.txt',
        type=str,
        help="The output file for resulting map",
    )

    dump_map_parser.add_argument(
        "--out",
        default=f'list_out.txt',
        type=str,
        help="The output file for resulting map",
    )

    dump_map_parser.add_argument(
        "--reduced-level",
        default=0,
        type=int,
        help="The reduced level to check if a rhyme can't be found for the original level",
    )

    dump_map_parser.add_argument(
        "--max-tries",
        default=4,
        type=int,
        help="The max tries to get a rhyme from that particular pronounciation",
    )

    dump_map_parser.add_argument(
        "--mode",
        default='w',
        type=str,
        help="Write mode, defaults to w, but could be 'a'. Don't make it r",
    )

    dump_map_parser.add_argument(
        "--shuffle",
        default=0,
        type=int,
        help="Whether or not to randomize the order of lists of each dictionary item",
    )

    dump_map_parser.add_argument(
        "--min-words",
        default=4,
        type=int,
        help="Minimum words a line needs to have to be printed",
    )

    dump_map_parser.add_argument(
        "--debug",
        default=0,
        type=int,
        help="debug val",
    )

    dump_map_parser.set_defaults(func=dump_map)


    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    # call python3 -m rhymetool
    cli()
