#!/usr/bin/env bash

# generate-identifier.sh - given a file name, return a key


# sanity check
if [[ -z $1 ]]; then

	echo "Usage: $0 <filename>" >&2
	exit
fi

FILE=$1

FILENAME=$(basename -- "$FILE")
IDENTIFIER="${FILENAME%.*}"

# output and done
echo "$IDENTIFIER"
exit
