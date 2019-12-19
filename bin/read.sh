#!/usr/bin/env bash

# read.sh - given an input file and an output directory, do sets of natural language processing

# Eric Lease Morgan <eric_morgan@infomotions.com>
# December 11, 2019 - first investigations; "Happy Birthday, Lincoln"


# configure
MAKEFILESYSTEM='./bin/make-filesystem.sh'
CACHEFILE='./bin/cache-file.sh'
FILE2TXT='./bin/file2txt.py'
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

# initialize debugig
clear

# generate identifier
printf "Step #01 of 16: Generating identifier\n" >&2
IDENTIFIER=$( $GENERATEIDENTIFIER "$FILE" )

# create file system
printf "Step #02 of 16: Making file system\n" >&2
$MAKEFILESYSTEM "$DIRECTORY"

# cache file
printf "Step #03 of 16: Caching original file\n" >&2
$CACHEFILE "$FILE" "$DIRECTORY"

# transform file into plain text
printf "Step #04 of 16: Transforming file into plain text\n" >&2
$FILE2TXT "$FILE" > "$DIRECTORY/$IDENTIFIER-text.txt"

# re-configure
FILE="$DIRECTORY/$IDENTIFIER-text.txt"

# fork off various extractions; bibliographics( identifier, author, title, date, summary, words, flesch, etc )
printf "Step #05 of 16: Extracting bibliographics\n" >&2
$TXT2BIB "$FILE" > "$DIRECTORY/$IDENTIFIER-bib.tsv" &

# unigrams
printf "Step #06 of 16: Extracting unigrams\n" >&2
$TXT2NGRAMS 1 "$FILE" > "$DIRECTORY/$IDENTIFIER-unigrams.tsv" &

# bigrams 
printf "Step #07 of 16: Extracting bigrams\n" >&2
$TXT2NGRAMS 2 "$FILE" > "$DIRECTORY/$IDENTIFIER-bigrams.tsv" &

# trigrams
printf "Step #08 of 16: Extracting trigrams\n" >&2
$TXT2NGRAMS 3 "$FILE" > "$DIRECTORY/$IDENTIFIER-trigrams.tsv" &

# email addresses
printf "Step #09 of 16: Extracting email addresses\n" >&2
$TXT2ADR "$FILE" > "$DIRECTORY/$IDENTIFIER-email.tsv" &

# URLs
printf "Step #10 of 16: Extracting URLs\n" >&2
$TXT2URLS "$FILE" > "$DIRECTORY/$IDENTIFIER-urls.tsv" &

# named-entities
printf "Step #11 of 16: Extracting named-entities\n" >&2
$TXT2ENT "$FILE" > "$DIRECTORY/$IDENTIFIER-entities.tsv" &

# parts-of-speech
printf "Step #12 of 16: Extracting parts-of-speech\n" >&2
$TXT2POS "$FILE" > "$DIRECTORY/$IDENTIFIER-pos.tsv" &

# keywords
$TXT2KEYWORDS "$FILE" > "$DIRECTORY/$IDENTIFIER-keywords.tsv" &

# various grammers; subjects-verbs-objects
printf "Step #13 of 16: Extracting subjects-verbs-objects\n" >&2
$TXT2SVO "$FILE" > "$DIRECTORY/$IDENTIFIER-svo.tsv" &

# noun-verb

# adjective noun

# wait until extractions are done
printf "Step #14 of 16: Waiting for extractions to finish\n" >&2
wait

# initialize database
#printf "Step #15 of 16: Initializing database\n" >&2
#$INITIALIZEDATABASE "$DIRECTORY" "$IDENTIFIER"

# write narrative report(s)
printf "Step #15 of 16: Generating narrative reports\n" >&2
$ANALYSIS2TXT "$DIRECTORY" "$IDENTIFIER" > "$DIRECTORY/$IDENTIFIER-reader.txt" &
$ANALYSIS2HTM "$DIRECTORY" "$IDENTIFIER" > "$DIRECTORY/$IDENTIFIER-reader.htm" &

# done
printf "Done\n" >&2
exit
