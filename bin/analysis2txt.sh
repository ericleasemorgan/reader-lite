#!/usr/bin/env bash

# analysis2txt.sh - given a Reader Lite directory ("study carrel") and an identifier, output a simple narrative

# Eric Lease Morgan <eric_morgan@infomotions.com>
# December 11, 2019 - first cut; "Happy Birthday, Lincoln!"


# sanity check
if [[ -z $1 || -z $2 ]]; then

	echo "Usage: $0 <directory> <identifier>" >&2
	exit
	
fi

# get input
DIRECTORY=$1
IDENTIFIER=$2

# do the work and done
echo "This is a simple narrative report against the content in $DIRECTORY and identified by $IDENTIFIER. --ELM (December 11, 2019)"
exit
