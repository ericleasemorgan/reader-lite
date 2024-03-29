#!/usr/bin/env bash

# txt2email.sh - given a file name, output a list of email addresses

# usage: find carrels/word2vec/txt -name '*.txt' -exec ./bin/txt2adr.sh {} \;

# Eric Lease Morgan <emorgan@nd.edu>
# (c) University of Notre Dame and distributed under a GNU Public License

# June 26, 2018 - first cut


# sanity check
if [[ -z "$1" ]]; then
	echo "Usage: $0 <file>" >&2
	exit
fi

# get input
FILE=$1

# extract the data
RECORDS=$( cat "$FILE" | grep -i -o '[A-Z0-9._%+-]\+@[A-Z0-9.-]\+\.[A-Z]\{2,4\}' )

SIZE=${#RECORDS} 
if [[ $SIZE > 0 ]]; then

	# proces each item in the data
	printf "id\taddress\n"
	while read -r RECORD; do
		printf "$LEAF\t$RECORD\n"
	done <<< "$RECORDS"

fi
