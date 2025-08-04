#!/usr/bin/env python

# server.py - a Web interface to the index of Distant Reader sentence embeddings lite

# Eric Lease Morgan <emorgan@nd.edu>
# (c) Infomotions, LLC; distributed under a GNU Public License

# August 3, 2025 - first investigations; rooted in the non-lite version so I can share and demonstrate


# configure
EMBEDDER      = 'nomic-embed-text'
LLM           = 'llama2'
CACHEDRESULTS = './etc/cached-results.txt'
CACHEDCARREL  = './etc/cached-carrel.txt'
CACHEDQUERY   = './etc/cached-query.txt'
CACHEDCITES   = './etc/cached-cites.txt'
SYSTEMPROMPT  = './etc/system-prompt.txt'

# require
from flask                    import Flask, render_template, request
from math                     import exp
from ollama                   import embed, generate
from pandas                   import DataFrame, read_csv, array
from pathlib                  import Path
from re                       import sub
from scipy.signal             import argrelextrema
from sklearn.metrics.pairwise import cosine_similarity
from sqlite_vec               import load
from sqlite3                  import connect
from struct                   import pack
from typing                   import List
import numpy                  as     np


# the system's work horse
def search( carrel, query, depth ) :

	# configure
	COLUMNS  = [ 'titles', 'items', 'sentences', 'distances' ]
	SELECT   = "SELECT title, item, sentence, VEC_DISTANCE_L2(embedding, ?) AS distance FROM sentences ORDER BY distance LIMIT ?"
	DATABASE = 'sentences.db'
	LIBRARY  = 'carrels'

	# initialize
	library  = Path( LIBRARY )
	
	# cache the carrel and query
	with open( Path( CACHEDQUERY ), 'w' )  as handle : handle.write( query )

	# initialize some more
	database = connect( library/carrel/DATABASE, check_same_thread=False )
	database.enable_load_extension( True )
	load( database )

	# vectorize query and search; get a set of matching records
	query   = embed( model=EMBEDDER, input=query ).model_dump( mode='json' )[ 'embeddings' ][ 0 ]
	records = database.execute( SELECT, [ serialize( query ), depth ] ).fetchall()
	
	# process each result; create a list of sentences
	sentences = []
	for record in records :
	
		# parse
		title    = record[ 0 ]
		item     = record[ 1 ]
		sentence = record[ 2 ]
		distance = record[ 3 ]
		
		# update
		sentences.append( [title, item, sentence, distance ] )
	
	# create a dataframe of the sentences and sort by title
	sentences = DataFrame( sentences, columns=COLUMNS )
	sentences = sentences.sort_values( [ 'titles', 'items' ] )

	# process/output each sentence; along the way, create a cache
	results = []
	cites   = []
	for index, result in sentences.iterrows() :
	
		# parse
		title    = result[ 'titles' ]
		item     = result[ 'items' ]
		sentence = result[ 'sentences' ]
		
		# update the caches
		results.append( sentence )
		cites.append( '\t'.join( [ title, str( item ) ] ) )
		
	# save the remaining caches
	with open( Path( CACHEDCITES ), 'w' )   as handle : handle.write( '\n'.join( cites ) )
	with open( Path( CACHEDRESULTS ), 'w' ) as handle : handle.write( '\n'.join( results ) )

	# format the result and done
	results = ' '.join( results )
	return( results )
	

# serializes a list of floats into a compact "raw bytes" format; makes things more efficient?
def serialize( vector: List[float]) -> bytes : return pack( "%sf" % len( vector ), *vector )

def rev_sigmoid( x:float )->float : return ( 1 / ( 1 + exp( 0.5*x ) ) )

def activate_similarities( similarities:np.array, p_size=10 )->np.array :
        
        # To create weights for sigmoid function we first have to create space. P_size will determine number of sentences used and the size of weights vector.
        x = np.linspace( -10, 10, p_size )
 
        # Then we need to apply activation function to the created space
        y = np.vectorize(rev_sigmoid) 
 
        # Because we only apply activation to p_size number of sentences we have to add zeros to neglect the effect of every additional sentence and to match the length ofvector we will multiply
        activation_weights = np.pad(y(x),(0,similarities.shape[0]-p_size))
 
        ### 1. Take each diagonal to the right of the main diagonal
        diagonals = [similarities.diagonal(each) for each in range(0,similarities.shape[0])]
 
        ### 2. Pad each diagonal by zeros at the end. Because each diagonal is different length we should pad it with zeros at the end
        diagonals = [np.pad(each, (0,similarities.shape[0]-len(each))) for each in diagonals]
 
        ### 3. Stack those diagonals into new matrix
        diagonals = np.stack(diagonals)

        ### 4. Apply activation weights to each row. Multiply similarities with our activation.
        diagonals = diagonals * activation_weights.reshape(-1,1)
 
        ### 5. Calculate the weighted sum of activated similarities
        activated_similarities = np.sum(diagonals, axis=0)

        # done
        return( activated_similarities )


# configure
server = Flask(__name__)

# home
@server.route( "/" )
def home() : return render_template('home.htm' )


# search
@server.route( "/search/" )
def searchSimple() :

	# get the cached carrel
	carrel = open( Path( CACHEDCARREL ) ).read().split( '\t' )
		
	# get input
	query = request.args.get('query', '')
	depth = request.args.get('depth', '')

	if not query or not depth : return render_template('search-form.htm', carrel=carrel )
		
	# search
	results = search( carrel[ 0 ], query, depth )
	
	# done
	return render_template('search.htm', results=results)


# elaborate
@server.route( "/elaborate/" )
def elaborate() :

	# configure
	PROMPT = 'Answer the question "%s" and use only the following as the source of the answer: %s'

	# get input
	question = request.args.get('question', '')
	
	if not question : return render_template('elaborate-form.htm' )

	# initialize
	context = open( CACHEDRESULTS ).read()
	system  = open( SYSTEMPROMPT ).read()
	prompt  = ( PROMPT % ( question, context ))

	# do the work
	result = generate( LLM, prompt, system=system )

	# reformat the results
	response = sub( '\n\n', '</p><p>', result[ 'response' ] ) 
	response = '<p>' + response + '</p>'

	# done
	#return( response )
	return render_template('elaborate.htm', results=response )


# summarize
@server.route("/summarize/")
def summarize() :

	# configure
	PROMPT = 'Summarize the following: %s'

	# initialize
	context = open( CACHEDRESULTS ).read()
	system  = open( SYSTEMPROMPT ).read()
	prompt  = ( PROMPT % ( context ) )

	try: results = generate( LLM, prompt, system=system )
	except ConnectionError : exit( 'Ollama is probably not running. Start it. Otherwise, call Eric.' )
	
	response = sub( '\n\n', '</p><p>', results[ 'response' ] ) 
	results = '<p>' + response + '</p>'

	#return( response )
	return render_template('summarize.htm', results=results )


# persona
@server.route("/persona/")
def persona() :

	# configure
	PERSONAS = [ 'a child in the second grade', 'a child in the eigth grade', 'a high school valedictorian', 'a sophmoric college student', 'a helpful librarian', 'a university professor', 'an erudite scholar' ]
	PREFIX   = 'You are '
	SUFFIX   = '.'

	# get input
	persona = request.args.get( 'persona', '' )
	if not persona : return render_template('persona-form.htm', personas=PERSONAS )

	# save
	with open( Path( SYSTEMPROMPT ), 'w' )   as handle : handle.write( PREFIX + persona + SUFFIX )
	return render_template('persona.htm', persona=persona )
	


# carrel
@server.route("/choose/")
def choose() :

	# configure
	CARRELS = './etc/carrels.csv'
	
	carrels = read_csv( Path( CARRELS ) )
	carrels = [ row.tolist() for index, row in carrels.iterrows() ]	
	
	# get the cached carrel
	selected = open( Path( CACHEDCARREL ) ).read().split( '\t' )[ 0 ]

	# get input
	carrel = request.args.get( 'carrel', '' )
	if not carrel : return render_template('carrel-form.htm', carrels=carrels, selected=selected )
	
	# split the input into an array; kinda dumb
	carrel = carrel.split( '--' )
			
	# save
	with open( Path( CACHEDCARREL ), 'w' )   as handle : handle.write( '\t'.join( carrel ) )
	return render_template('carrel.htm', carrel=carrel )
	


# format
@server.route("/format/")
def format() :

	# configure
	PSIZE = 16

	# initialize
	sentences = open( CACHEDRESULTS ).read().splitlines()

	# vectorize and activated similaritites; for longer sentences increase the value of PSIZE
	embeddings = embed( model=EMBEDDER, input=sentences ).model_dump( mode='json' )[ 'embeddings' ]

	#try : similarities = activate_similarities( cosine_similarity(embeddings), p_size=PSIZE )
	#except ValueError as error : exit( "Number of sentences too small. If this error continues, call Eric.\n" )

	similarities = activate_similarities( cosine_similarity(embeddings), p_size=PSIZE )

	minmimas = argrelextrema( similarities, np.less, order=2 )

	# Get the order number of the sentences which are in splitting points
	splits = [ minmima for minmima in minmimas[ 0 ] ]

	# Create empty string
	text = ''
	for index, sentence in enumerate( sentences ) :
	
		# Check if sentence is a minima (splitting point)
		if index in splits : text += f'\n\n{sentence} '
		else               : text += f'{sentence} '

	# do the tiniest bit of normalization
	text = sub( ' +', ' ', text ) 
	text = '<p>' + sub( '\n\n', '</p><p>', text ) + '</p>'

	# done
	#return( sentences )
	return render_template('format.htm', results=text )


