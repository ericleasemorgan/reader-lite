#!/usr/bin/env bash

# read.sh - given an input file and an output directory, do sets of natural language processing

# Eric Lease Morgan <eric_morgan@infomotions.com>
# December 11, 2019 - first investigations; "Happy Birthday, Lincoln"


# configure
MAKEFILESYSTEM='./bin/make-filesystem.sh'
CACHEFILE='./bin/cache-file.sh'
FILE2TXT='./bin/file2txt.sh'
GENERATEIDENTIFIER='./bin/generate-identifier.sh'
ANALYSIS2TXT='./bin/analysis2txt.sh'
ANALYSIS2HTM='./bin/analysis2htm.sh'
TXT2ENT='./bin/txt2ent.py'
TXT2BIB='./bin/txt2bib.py'
TXT2KEYWORDS='./bin/txt2keywords.py'
TXT2POS='./bin/txt2pos.py'
TXT2SVO='./bin/txt2svo.py'
TXT2ADR='./bin/txt2adr.sh'
TXT2NGRAMS='./bin/txt2ngrams.py'
TXT2URLS='./bin/txt2urls.sh'

SCHEMA='./etc/schema.sql'
INITIALIZEDATABASE='./bin/db-initialize.sh'

# sanity check
if [[ -z $1 || -z $2 ]]; then
	echo "Usage: $0 <input file> <output directory>" >&2
	exit
fi

# get input
FILE=$1
DIRECTORY=$2

# generate identifier
printf "Generating identifier             \r" >&2
IDENTIFIER=$( $GENERATEIDENTIFIER "$FILE" )

# create file system
printf "Making file system                \r" >&2
$MAKEFILESYSTEM "$DIRECTORY"

# cache file
printf "Caching original file             \r" >&2
$CACHEFILE "$FILE" "$DIRECTORY"

# transform file into plain text
printf "Transforming file into plain text \r" >&2
$FILE2TXT "$FILE" > "$DIRECTORY/$IDENTIFIER-text.txt"

# fork off various extractions; bibliographics( identifier, author, title, date, summary, words, flesch, etc )
$TXT2BIB "$FILE" > "$DIRECTORY/$IDENTIFIER-bib.tsv" &

# unigrams
$TXT2NGRAMS 1 "$FILE" > "$DIRECTORY/$IDENTIFIER-unigrams.tsv" &

# bigrams 
$TXT2NGRAMS 2 "$FILE" > "$DIRECTORY/$IDENTIFIER-bigrams.tsv" &

# trigrams
$TXT2NGRAMS 3 "$FILE" > "$DIRECTORY/$IDENTIFIER-trigrams.tsv" &

# email addresses
$TXT2ADR "$FILE" > "$DIRECTORY/$IDENTIFIER-email.tsv" &

# URLs
$TXT2URLS "$FILE" > "$DIRECTORY/$IDENTIFIER-urls.tsv" &

# named-entities
$TXT2ENT "$FILE" > "$DIRECTORY/$IDENTIFIER-entities.tsv" &

# parts-of-speech
$TXT2POS "$FILE" > "$DIRECTORY/$IDENTIFIER-pos.tsv" &

# keywords
$TXT2KEYWORDS "$FILE" > "$DIRECTORY/$IDENTIFIER-keywords.tsv" &

# various grammers; subjects-verbs-objects
$TXT2SVO "$FILE" > "$DIRECTORY/$IDENTIFIER-svo.tsv" &

# noun-verb

# adjective noun

# wait until extractions are done
printf "Waiting for extractions to finish \r" >&2
wait

# initialize database
printf "Initializing database             \r" >&2
$INITIALIZEDATABASE "$DIRECTORY" "$IDENTIFIER"

# write narrative report(s)
printf "Generating narrative reports      \r" >&2
$ANALYSIS2TXT "$DIRECTORY" "$IDENTIFIER" > "$DIRECTORY/$IDENTIFIER-reader.txt" &
$ANALYSIS2HTM "$DIRECTORY" "$IDENTIFIER" > "$DIRECTORY/$IDENTIFIER-reader.htm" &

# done
printf "Done                              \n" >&2
exit
