#!/usr/bin/env bash

# cache-file.sh - given a file name and a directory, copy the file to the directory

# Eric Lease Morgan <eric_morgan@infomotions.com>
# December 11, 2019 - first cut; "Happy Birthday, Lincoln!"


CACHE='cache'

# sanity check
if [[ -z $1 || -z $2 ]]; then

	echo "Usage: $0 <file> <directory>" >&2
	exit
	
fi

# get input
FILE=$1
DIRECTORY=$2

# do the work and done
cp "$FILE" "$DIRECTORY"
cp "$FILE" "$DIRECTORY/../$CACHE"
exit
