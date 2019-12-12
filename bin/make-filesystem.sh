#!/usr/bin/env bash

# make-filesystem.sh - fill a given directory with a set of subdirectories

# Eric Lease Morgan <eric_morgan@infomotions.com>
# December 11, 2019 - first cut; "Happy Birthday, Lincoln!"


# sanity check
if [[ -z $1 ]]; then

	echo "Usage: $0 <directory>" >&2
	exit

fi

# get input
DIRECTORY=$1

mkdir -p "$DIRECTORY"
exit
