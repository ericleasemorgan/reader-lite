#!/usr/bin/env bash

# analysis2htm.sh - given a Reader Lite directory ("study carrel") and an identifier, output a simple HTML narrative

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
echo "<html><head><title>$IDENTIFIER</title></head><body><h1>$IDENTIFIER</h1><p>This is a simple HTML narrative report against the content in $DIRECTORY and identified by $IDENTIFIER. --ELM (December 11, 2019)</p></body></html>"
exit
