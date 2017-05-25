# miniprojects
A description of weekly projects completed throughout Spring 2017 for The Data Incubator Fellowship. Linked detailed iPython walk-throughs forthcoming. Please feel free to inquire for more detailed code, out of respect for current and future fellows no direct answers will be posted.

## (1) Using Graph and Network Theory To Investigate the NYC Social Elite

<a href="http://www.newyorksocialdiary.com/">The New York Social Diary</a> provides a fascinating lens onto New York's socially well-to-do. As shown in <a href="http://www.newyorksocialdiary.com/party-pictures/2014/holiday-dinners-and-doers">this report of a recent holiday party</a>, almost all the photos have annotated captions labeling their subjects. We can think of this as implicitly implying a social graph: there is a connection between two individuals if they appear in a picture together.

In this project, I investigate these connections between the NYC elite. I determine who the most socially popular and influential people are, as well as pairs with the strongest connections. (The latter analysis picks up the obvious  -- marriages and family ties -- as well as connections not readily apparent, such as affairs and infidelities.) Methodologically this was completed by scraping all website pages for photo captions, total number of valid captions was 93,093. These captions were then parsed to return unique names, revealing a total of 113,031 names in the social network. Finally, I performed the graph analysis using networkx. The graph visualization can be viewed in the screenshot below.

_TOOLS USED_: Python - BeutifulSoup (webscrape), regex (cleaning website data), networkx (graph/viz)

![network graph](https://github.com/beilmanmich/datasci_projects/blob/master/network_graph.png)

## (2) Using PostGreSQL and Pandas To Investigate NYC Restaurants

The city of New York inspect roughly 24,000 restaraunts a year and assigns a grade to restaurants after each inspection, over a decade this creates a public dataset of 531,935 records. I used PostGreSQL and Pandas to parse and analyze a decade worth of NYC Restaurant Inspections data. I extracted different slices -- determining the grade distribution by zipcode, borough, and cuisine. I also found which cuisines tended to have a disproportionate number of which violations.

![map image](https://github.com/beilmanmich/datasci_projects/blob/master/carto_map.png)

A map view of scores by zipcode, where higher intensity reds are equivalent to higher scores, via cartodb.

_TOOLS USED_: Pandas, numPy, psql, SQLAlchemy, matplotlib, seaborn

## (3) Using Semi-Structured Data to Predict Yelp Ratings

I attempted to predict a new venue's popularity from information available when the venue opens. <a href="https://www.yelp.com/developers/documentation/v2/business">The dataset</a> contains unstructured meta data about each venue (city, latitude/longitude, category descriptions, etc), and a star rating. After data munging and feature variable creation, I used Python's [Scikit Learn](http://scikit-learn.org/stable/modules/linear_model.html) [libraries](http://scikit-learn.org/stable/modules/neighbors.html) to create transformations to allow us to model feature data, since the data is unstructured this required json parsing to create flattened nested dictionaries. I then used Scikit Learn to create training and test data to develop several different Machine Learning algorithms.

The predictive algorithms revealed three weak predictors (city, lat/long, attributes) and one fair predictor (categories). I ultimately used a [FeatureUnion](http://scikit-learn.org/stable/modules/generated/sklearn.pipeline.FeatureUnion.html) approach and created an [ensemble regressor](http://scikit-learn.org/stable/modules/ensemble.html) for the final model. In Python after creating our custom data [transformers](http://scikit-learn.org/stable/modules/generated/sklearn.base.TransformerMixin.html), the final [pipeline](http://scikit-learn.org/stable/modules/generated/sklearn.pipeline.Pipeline.html) looks like this:

```
pipeline = Pipeline([
  ('features', FeatureUnion([
    ('city', Pipeline([
      ('cst', ColumnSelectTransformer(columns=['city'])),
      ('classifier', ModelTransformer(CityMeanEstimator(),'city class'))
    ])),
    ('neighborhod', Pipeline([
      ('lat lon cst', ColumnSelectTransformer(columns=['latitude','longitude'])),
      ('nearest neighbors', ModelTransformer(neighbors.KNeighborsRegressor(n_neighbors=94),'neigh class'))
    ])),
    ('category', Pipeline([
      ('cat cst', ColumnSelectTransformer(columns=['categories'])),
      ('cat_transformer', CategoryTransformer()),
      ('cat vectorizer', DictVectorizer()),
      ('cat tfidf', TfidfTransformer()),
      ('classifier', ModelTransformer(linear_model.LinearRegression(),'cat class'))
    ])),
    ('attributes', Pipeline([
      ('att cst', ColumnSelectTransformer(columns=['attributes'])),
      ('att_transformer', AttributesTransformer()),
      ('att vectorizer', DictVectorizer()),
      ('att tfidf',TfidfTransformer()),
      ('classifier', ModelTransformer(linear_model.LinearRegression(),'att class'))
    ]))
  ])),
  ('final classifier', linear_model.LinearRegression())
])
```

One pipeline is required for each variable given the unstructured nature of the Yelp data, the transformers fit our data to interact with SciKit Learn. After training and scoring the pipelined model...

```
pipeline.fit(data_train,stars_train)
print pipeline.predict(data_test[:10])
print stars_test[:10].values
score = pipeline.score(data_test,stars_test)
print score
preds = pipeline.predict(data_test)
print 'Ensemble Model RMSE: ', mean_squared_error(stars_test,preds)**0.5
print 'Ensemble Model MAE: ', mean_absolute_error(stars_test,preds)
print 'Ensemble Model R^2: ', r2_score(stars_test,preds)
```
----
```
[ 3.6706147   3.99940255  3.97274505  3.83213406  3.79369298  3.6706147
  3.31230485  3.41025357  3.58801357  3.72347899]
[ 5.   4.   4.5  4.5  3.5  5.   4.   3.5  4.   5. ]
0.0789390332998
Ensemble Model RMSE:  0.850048323084
Ensemble Model MAE:  0.675276799799
Ensemble Model R^2:  0.0789390332998
```

We have a model that predicts a star rating based on city, lat/lon, category and venue attributes (178 possible). Despite our efforts our model has relatively low predictive power, as indicated by the low R^2 value.

_TOOLS USED_: numpy, seaborn, pandas, dill, sklearn - KNeighborsRegressor, linear_model, feature_extraction, neighbors, cross_validation, grid_search, train_test_split, numpy, matplotlib

## (4) Analyzing Wikipedia via MapReduce

I scraped the entire English Wikipedia to determine its <a href="https://github.com/beilmanmich/datasci_projects/blob/master/MapReduce/top_100_words.py">most frequently used words</a>. I then collected <a href="https://github.com/beilmanmich/datasci_projects/blob/master/MapReduce/linkstats.py">link statistics</a> (mean, standard deviations, percentiles) for the unique links on each page, and <a href="https://github.com/beilmanmich/datasci_projects/blob/master/MapReduce/double_link_stats.py">found the top linked concepts</a> by examining doublelinks.

Distributed computing is ideal for these types of tasks, as they allow for the distributed processing of large data sets across clusters of computers using simple programming models. <a href="http://hadoop.apache.org/">Hadoop</a> grew out of an open-source search engine called Nutch, developed by Doug Cutting and Mike Cafarella. Back in the early days of the Internet, the pair wanted to invent a way to return web search results faster by distributing data and calculations across different computers so multiple tasks could be executed at the same time. Hadoop Distributed File System (HDFS) is a distributed file system that provides high-throughput access (multi-terabyte data-sets) to application data, a Hadoop YARN is the framework for job scheduling and cluster resource management within the HDFS environment. With these two in place, one can run [MapReduce](https://hadoop.apache.org/docs/r1.2.1/mapred_tutorial.html) jobs, which splits input data into independent chunks which are process by the _map tasks_ across large _clusters_ (thousands of nodes). Utilizing the [MRJob](https://github.com/Yelp/mrjob) Python package developed at Yelp, one can write MapReduce jobs in Python.

_TOOLS USED_: Hadoop/HDFS, MapReduce, MRJob, XML, BeutifulSoup, Custom Builty XML Parser, Google Cloud Platform, AWS

## (5) DataViz

This project required fellows to create a webhosted interactive data visualization tool. My project analyzed historical project data from DonorsChoose.org, a dataset of 950,000 records (~8GB). More information on my project can be <a href="https://github.com/beilmanmich/donors_dashboard">found here</a>.

![data viz](https://github.com/beilmanmich/datasci_projects/blob/master/viz_demo.gif)

_TOOLS USED_: d3.js, dc.js, html, css, java, Flask, MongoDB, Numpy, Heroku

## (6) Using Time Series Machine Learning to Predict the Weather

Time series prediction presents its own challenges which are different from machine-learning problems. As with many other classes of problems, there are a number of common features in these predictions. In this project, I built two time series models to predict weather in a given city at a given time, based on historical trends. Seasonal features are nice because they are relatively safe to extrapolate into the future. There are two ways to handle seasonality.

The simplest (and perhaps most robust) is to have a set of indicator variables. That is, make the assumption that the temperature at any given time is a function of only the month of the year and the hour of the day, and use that to predict the temperature value. A more complex approach is to fit our model to the time series curve. Since we know that temperature is roughly sinusoidal, we know that a resonable model might be:

![math eqn](https://latex.codecogs.com/gif.latex?%24%24%20y_t%20%3D%20y_0%20%5Csin%5Cleft%282%5Cpi%5Cfrac%7Bt%20-%20t_0%7D%7BT%7D%5Cright%29%20&plus;%20%5Cepsilon%20%24%24)

where ![alt text](https://latex.codecogs.com/gif.latex?%24k%24) and ![alt text](https://latex.codecogs.com/gif.latex?%24t_0%24) are parameters to be learned and ![alt text](https://latex.codecogs.com/gif.latex?%24T%24) is one year for seasonal variation.  While this is linear in ![alt text](https://latex.codecogs.com/gif.latex?%24y_0%24), it is not linear in ![alt text](https://latex.codecogs.com/gif.latex?%24t_0%24). However, we know from <a href="https://en.wikipedia.org/wiki/Fourier_analysis">Fourier analysis</a>, that the above is equivalent to

![math eqn2](https://latex.codecogs.com/gif.latex?%24%24%20y_t%20%3D%20A%20%5Csin%5Cleft%282%5Cpi%5Cfrac%7Bt%7D%7BT%7D%5Cright%29%20&plus;%20B%20%5Ccos%5Cleft%282%5Cpi%5Cfrac%7Bt%7D%7BT%7D%5Cright%29%20&plus;%20%5Cepsilon%20%24%24)

which is linear in ![alt text](https://latex.codecogs.com/gif.latex?%24A%24) and ![alt text](https://latex.codecogs.com/gif.latex?%24B%24).

_TOOLS USED_: Pandas, sklearn - BaseEstimator, TransformerMixin,  DictBectorizer, Pipeline, LogisticRegression, KNeighborsRegressor

## (7) Using NLP to Predict Yelp Ratings

I then returned to the Yelp Data to explore how much information was contained in the review texts, whether they could more accurately predict ratings. This project allowed a rich dataset to practice <a href="http://www.kdnuggets.com/2015/12/natural-language-processing-101.html">natural language processing</a> (NLP) as the unstructured data contains the text record of a written yelp review. Since I was working with over one million reviews and a design matrix of over a million feature-words, scalability was an overriding factor during model selection, especially since the model had to fit within Heroku's memory constraints. However, the predictive power of even a basic out-of-the-box ridge regression was magnitudes greater than that of the models in the previous section (yielding a score of over .6).

_TOOLS USED_: nltk, nltk.tokenize, nltk.WordNetLemmatizer, Textblob, dill, numpy, pandas, seaborn, matplotlib.pylab, sklearn - linear_models, externals, joblib, cross_validation, grid_searchCV, FeatureUnion, Pipeline, CountVectorizer, HashingVectorizer, Tfidf, Tfidf_vectorizer

## (8) Using ML to Categorize Music Samples

Music offers an extremely rich and interesting playing field. The objective of this miniproject is to develop models that are able to recognize the genre of a musical piece (electronic, folkcountry, jazz, raphiphop, rock), first from pre-computed features and then working from the raw waveform (files with 5-10 seconds of a music sample). This is a typical example of a classification problem on time series data. 

_TOOLS USED_: numpy, scipy, Librosa, audioread, audioop, pandas, dill, sklearn - preprocessing, ensemble, LabelEncoder, DecisionTreeClassifier, StandardScaler, normalize, PCA, RandomForestClassifier, Pipeline, SVM, SVC

## (9) SparkOverflow - Using Spark to Analyze StackOverflow Data

StackOverflow is a collaboratively edited question-and-answer site originally focused on programming topics. Because of the variety of features tracked, including a variety of feedback metrics, it allows for some open-ended analysis of user behavior on the site.

StackExchange (the parent organization) provides an anonymized <a href="https://archive.org/details/stackexchange">data dump</a>, this project  used Spark to perform data manipulation, analysis, and machine learning on this dataset. Similar to the MapReduce project, this is an ideal use for distributed computing. <a href="https://spark.apache.org/">Spark</a> is Hadoop's bigger, better, stronger, faster cousin -- and runs with the ability to cache, significantly increasing the speed over Hadoop.  Using [PySpark](https://spark.apache.org/docs/0.9.0/python-programming-guide.html) and [Google Cloud Platform](https://cloud.google.com/), I was able to perform distributed computing on a massive 30GB dataset of unstructured XML, spanning 341 seperate files. Using Python's Scala API my example workflow is as follows:

(a) Edit source code in your main.py file, classes in a separate classes.py (Class definitions need to be written in a separate file and then included at runtime.)
(b) Run locally on a chunk using eg. `$SPARK_HOME/bin/spark-submit --py-files src/classes.py src/main.py data/stats results/stats/`
(c) Run on GCP once your testing and development are done. 

_TOOLS USED_: Spark - PySpark and Scala, HDFS, Google Cloud Platform

----
## A summary of my performance across projects

A grade of 1.0 indicates no variance from the referenced solution outpout and performance, a grade of 0.9 is needed throughout. _(ETL / DW = Extract, Transact, Load & DataWarehousing; ML = Machine Learning, VIZ = DataViz, DC = Distributed Computing, AML = Advanced Machine Learning)_

| Week | Category 	| Project | Avg Grade |
| --- | -------------:| -----:|  -----:|
|1 |	ETL / DW	|1, 2 |	.984 |
|2 | ML	| 3	|.968|
|3|	DC	|4|	.978|
|4|	VIZ|	5|	1.0 |
|5|	AML|	6|	.963|
|6|	AML|	7|	1.071|
|7|	AML|	8|	.956|
|8|	DC|	9|	.994|

My Gradebook:

![grade book](https://github.com/beilmanmich/datasci_projects/blob/master/GradeBook_TDI.png)
