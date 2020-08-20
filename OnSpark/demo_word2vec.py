#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import print_function

from pyspark import SparkContext

from pyspark.mllib.feature import Word2Vec

if __name__ == "__main__":
    sc = SparkContext(appName="Word2VecExample")  # SparkContext

    inp = sc.textFile("/Users/tung/Documents/spark-2.4.3/data/mllib/sample_lda_data.txt").map(lambda row: row.split(" "))

    word2vec = Word2Vec()
    model = word2vec.fit(inp)

    synonyms = model.findSynonyms('1', 5)

    for word, cosine_distance in synonyms:
        print("{}: {}".format(word, cosine_distance))

    sc.stop()
