import pandas as pd
import numpy as np
import operator
from scipy import spatial

# load up every rating in the data set into a Pandas DataFrame

r_cols = ['user_id', 'movie_id', 'rating']
ratings = pd.read_csv('u.data', sep='\t', names=r_cols, usecols=range(3))
print(ratings.head(10))

# group everything by movie ID, and compute the total number of ratings (each movie's popularity) and the average rating for every movie

movieProperties = ratings.groupby('movie_id').agg({'rating': [np.size, np.mean]})
print(movieProperties.head(10))

# create a new DataFrame that contains the normalized number of ratings.
# So, a value of 0 means nobody rated it, and a value of 1 will mean it's the most popular movie there is.

movieNumRatings = pd.DataFrame(movieProperties['rating']['size'])
movieNormalizedNumRatings = movieNumRatings.apply(lambda x: (x - np.min(x)) / (np.max(x) - np.min(x)))
print(movieNormalizedNumRatings.head(10))

# get the genre information from the u.item file.
# The way this works is there are 19 fields, each corresponding to a specific genre -
# a value of '0' means it is not in that genre, and '1' means it is in that genre.
# A movie may have more than one genre associated with it.
# puting together everything into one big Python dictionary called movieDict.
# Each entry will contain the movie name, list of genre values, the normalized popularity score, and the average rating for each movie.

movieDict = {}
with open(r'u.item', encoding = "ISO-8859-1") as f:
    temp = ''
    for line in f:
        #line.decode("ISO-8859-1")
        fields = line.rstrip('\n').split('|')
        movieID = int(fields[0])
        name = fields[1]
        genres = fields[5:25]
        genres = map(int, genres)
        movieDict[movieID] = (name, np.array(list(genres)), movieNormalizedNumRatings.loc[movieID].get('size'), movieProperties.loc[movieID].rating.get('mean'))

# define a function that computes the "distance" between two movies based on how similar their genres are,
# and how similar their popularity is.
def ComputeDistance(a, b):
    genresA = a[1]
    genresB = b[1]
    genreDistance = spatial.distance.cosine(genresA, genresB)
    popularityA = a[2]
    popularityB = b[2]
    popularityDistance = abs(popularityA - popularityB)
    return genreDistance + popularityDistance

# Just to make sure it works, we'll compute the distance between movie ID's 2 and 4
print(movieDict[2])
print(movieDict[4])
print(ComputeDistance(movieDict[2], movieDict[4]))

# compute the distance between some given test movie (Toy Story, in this example) and all of the movies in our data set.
# sort those by distance, and print out the K nearest neighbors

def getNeighbors(movieID, K):
    distances = []
    for movie in movieDict:
        if (movie != movieID):
            dist = ComputeDistance(movieDict[movieID], movieDict[movie])
            distances.append((movie, dist))
    distances.sort(key=operator.itemgetter(1))
    neighbors = []
    for x in range(K):
        neighbors.append(distances[x][0])
    return neighbors


K = 10
avgRating = 0
neighbors = getNeighbors(1, K)
for neighbor in neighbors:
    avgRating += movieDict[neighbor][3]
    print(movieDict[neighbor][0] + " " + str(movieDict[neighbor][3]))

avgRating /= K
print(avgRating)

