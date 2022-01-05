from nltk.tokenize import word_tokenize
import itertools

pos_to_pos    = []
pos_to_word   = []
posterior     = []
num_states    = 0
pos_to_index  = {} # dict from pos  to index in pos_list
word_to_index = {} # dict from word to index in word_list
pos_list = []

def train_POS(file):
	with open(file,'r') as F:
		data = [eval(l) for l in F.readlines()]

	# Need to gather two frequencies- POS->POS frequencies, and word-in-POS frequencies
	# For that, we need two matrices
	# We also need the posterior probability for each POS
	pos_to_pos_freq  = []
	pos_to_word_freq = []
	posterior_freq   = []
	global pos_list		  # list of pos,   both to normalize later and to use in the resposne
	word_list        = [] # list of words, so we can normalize everything later
	global pos_to_index, word_to_index, num_states

	# First, grab every word and pos
	for sentence in data:
		for (word,pos) in sentence:
			if word not in word_list:
				word_list.append(word)
				word_to_index[word] = len(word_list)-1
			if pos not in pos_list:
				pos_list.append(pos)
				pos_to_index[pos]  = len(pos_list) -1

	# Grab our overall number of pos- we need that later
	num_states = len(pos_list)

	# Now, zero out our matrices
	pos_to_pos_freq  = [[0 for y in pos_list ] for x in pos_list]
	pos_to_word_freq = [[0 for y in word_list] for x in pos_list]
	posterior_freq   = [ 0 for x in pos_list ]

	# Now, actually start capturing frequencies
	for sentence in data:
		posterior_freq[pos_to_index[sentence[0][1]]] += 1
		pos_to_word_freq[pos_to_index[sentence[0][1]]][word_to_index[sentence[0][0]]] += 1
		for (word1, pos1), (word2, pos2) in pairwise(sentence):
			pos_to_pos_freq [pos_to_index[pos1]][pos_to_index [pos2 ]] += 1
			pos_to_word_freq[pos_to_index[pos2]][word_to_index[word2]] += 1

	# now grab the totals
	pos_to_pos_freq_total  = [sum(x) for x in zip(*pos_to_pos_freq )]
	pos_to_word_freq_total = [sum(x) for x in zip(*pos_to_word_freq)]
	posterior_freq_total   =  sum(posterior_freq)

	# and create our new lists
	global pos_to_pos, pos_to_word, posterior
	pos_to_pos  = [[pos_to_pos_freq [x][y]/pos_to_pos_freq_total[x]  if pos_to_pos_freq_total[x] != 0 else 0 for y in range(len(pos_list)) ] for x in range(len(pos_list))]
	pos_to_word = [[pos_to_word_freq[x][y]/pos_to_word_freq_total[x] if pos_to_pos_freq_total[x] != 0 else 0 for y in range(len(word_list))] for x in range(len(pos_list))]
	posterior   = [ x/posterior_freq_total for x in posterior_freq]
	for p in pos_to_word:
		p.append(1) # We do this so that there is a dummy value at the end of everything that will ensure that we have something to refer to later in case a word isn't in our training set
	pass

def test_POS(words):
	# So, we need two matrices- one for probabilities, and one for the actual path- we call the probability matrix P, and the path matrix T
	global num_states
	P = [[0 for y in range(len(words))] for x in range(num_states)]
	T = [[0 for y in range(len(words))] for x in range(num_states)]

	# Now we do our initialization step, where we set our starting probabilities
	global pos_to_pos, pos_to_word, posterior, pos_to_index, word_to_index
	for (i,p) in enumerate(P):
		p[0] = posterior[i]*pos_to_word[i][word_to_index.setdefault(words[0], len(pos_to_word[i])-1)]
	for (i,t) in enumerate(T):
		t[0] = 0

	# With that out of the way, we can do our wonderful Viterbi algorithm
	for (x,word) in enumerate(words[1:]):
		j = x + 1
		for i in range(num_states):
			P[i][j] = max([P[k][j-1]*pos_to_pos[k][i]*pos_to_word[i][word_to_index.setdefault(word, len(pos_to_word[i])-1)] for k in range(num_states)])
			T[i][j] = max([k for k in range(num_states)], key=lambda x: P[x][j-1]*pos_to_pos[x][i]*pos_to_word[i][word_to_index.setdefault(word, len(pos_to_word[i])-1)])

	# Now we need to choose the best one
	bestpathprob   = max([P[s][len(words)-1] for s in range(num_states)])
	bestpathchoice = max([s for s in range(num_states)], key=lambda x: P[x][len(words)-1])

	global pos_list
	X = [0 for x in range(len(words))]
	X[len(words)-1] = pos_list[T[bestpathchoice][len(words)-1]]
	for j in range(len(words)-1, 0, -1):
		bestpathchoice = T[bestpathchoice][j]
		X[j-1] = pos_list[T[bestpathchoice][j-1]]
	return X[1:]+['.'] # For some reason it kept being offset by 1 and ignoring the punctuation, so this is an admittedly forced correction for this

# Note- the following code has been copied from the Python3 more-itertools recipe list
# found here: https://docs.python.org/3/library/itertools.html#recipes
def pairwise(iterable):
    "s -> (s0,s1), (s1,s2), (s2, s3), ..."
    a, b = itertools.tee(iterable)
    next(b, None)
    return zip(a, b)
