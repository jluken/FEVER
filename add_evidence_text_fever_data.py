#!/usr/local/bin/python3
import json
from os import listdir
from os.path import isfile, join
import io
import logging
import string
import sys
import plac
import codecs
import unicodedata

def norm(x):
  return unicodedata.normalize('NFC', x)

def split_wiki_sentences(lines, filename):
    """
    split the original sentences into an array of sentences
    returns a list of pairs (sentence, list of links)
    """
    sent_link_pairs = []
    
    orig = lines
    
    if (len(lines) == 0):
        return []
    
    for line in lines.split("\n"):
        if ('\t' not in line or len(line) == 0):
            return []
        n = len(sent_link_pairs)
        splittabs = line.split("\t")
        first = splittabs[0]
        try:
            sec = splittabs[1]
        except IndexError:
            logging.warning(repr(line), splittabs, n, len(orig))
            
        links = splittabs[2:]
        uniq_links = set(links)
        if (first == str(n)):
            sent_link_pairs.append((sec, uniq_links))
            
        else:
            if (line == ""):
                continue
            try: 
                # if indices doesn't match, append everything to previous item
                (lastsent, lastlink) = sent_link_pairs[-1] 
                (newsent, newlink) = (lastsent, lastlink.union(set(splittabs)))
                sent_link_pairs[-1] = (newsent, newlink)
            except:
                continue
        
    return sent_link_pairs

def read_wiki_data(wiki_dir):
    if wiki_dir[:-1] != '/':
        wiki_dir += '/'

    onlyfiles = [f for f in listdir(wiki_dir) if isfile(join(wiki_dir, f))]
    wikidict = dict()

    for file in onlyfiles:
        filewpath = wiki_dir + file
        sys.stderr.write('.')
        with open(filewpath, encoding="utf-8") as f:
            for (n, line) in enumerate(f):
                content = json.loads(line)
                pageid = norm(content["id"])
                lines = content["lines"]
                splitted = split_wiki_sentences(lines, pageid)
                wikidict[pageid] = splitted

    sys.stderr.write('\n')
    return wikidict

@plac.annotations(
    wiki_dir=('wiki file directory', 'positional', None, str),
    input_file=('the path of the file to process', 'positional', None, str),
    output_file=('the path of the file to process', 'positional', None, str)
    )

def main(wiki_dir, input_file, output_file):
    sys.stderr.write("Reading wiki data")
    wikidict = read_wiki_data(wiki_dir)

    sys.stderr.write("Processing " + input_file)
    with open(input_file, 'r', encoding='utf-8') as f, open(output_file, "w") as newfile: 
        for (i, line) in enumerate(f):
            if (i % 1000 == 0):
                sys.stderr.write('.')
            content = json.loads(line)
            label = content["label"]
            if (label == "NOT ENOUGH INFO"):
                newfile.write(json.dumps(content))
                newfile.write("\n")
                continue
            evidence_sets = content["evidence"]
            for evidence_set in evidence_sets:
                for ev in evidence_set:
                    page_id = norm(ev[2])
                    sent_id = ev[3]
                    
                    try:
                        page = wikidict[page_id]
                    except KeyError:
                        logging.error("Can't find wiki page for %s", page_id)
                        continue
                    
                    try:
                        (sent, links) = page[sent_id]
                    except:
                        logging.error("Can't find sentence %d at page %s, list len", sent_id, page_id, len(page))
                        continue
                        
                    ev.append(sent)
                    ev.append(list(links))
                
            
            newfile.write(json.dumps(content))
            newfile.write("\n")
            
    newfile.close()
    sys.stderr.write("\nfinish writing")


if __name__ == '__main__':
    plac.call(main)
