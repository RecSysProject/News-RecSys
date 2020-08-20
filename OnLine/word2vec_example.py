

from __future__ import print_function

from pyspark import SparkContext
# $example on$
from pyspark.mllib.feature import Word2Vec
# $example off$

if __name__ == "__main__":
    sc = SparkContext(appName="Word2VecExample")  # SparkContext

    # $example on$
    inp = sc.textFile("data/mllib/sample_lda_data.txt").map(lambda row: row.split(" "))

    word2vec = Word2Vec()
    model = word2vec.fit(inp)

    synonyms = model.findSynonyms('1', 5)

    for word, cosine_distance in synonyms:
        print("{}: {}".format(word, cosine_distance))
    # $example off$

    sc.stop()
