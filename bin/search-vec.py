#!/usr/bin/env python

# search.py - query a semantic (word2vec) index

# Eric Lease Morgan <eric_morgan@infomotions.com>
# October 17, 2018 - first documentation


# configure
N = 10

# require
from gensim.models import KeyedVectors
import sys

# sanity check
if len( sys.argv ) != 3 :
	sys.stderr.write( 'Usage: ' + sys.argv[ 0 ] + " <vectors> <word>\n" )
	exit()

# initialize
vectors = sys.argv[ 1 ]

# load the index
index = KeyedVectors.load( vectors, mmap = 'r' )

# search and output
for word, score in index.most_similar( positive = sys.argv[ 2 ], topn = N ) :
	print( "\t".join( [ word, str( score ) ] ) )

# done
exit()
