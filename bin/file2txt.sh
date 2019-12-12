#!/usr/bin/env bash

# file2txt.sh - given a file, output plain text

# Eric Lease Morgan <eric_morgan@infomotions.com>
# December 11, 2019 - first cut; "Happy Birthday, Lincoln!"


# sanity check
if [[ -z $1 ]]; then

	echo "Usage: $0 <file>" >&2
	exit

fi

# get input
FILE=$1

# do the work and done
cat $FILE
exit