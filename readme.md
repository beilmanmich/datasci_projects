# miniprojects
A description of weekly projects completed Spring 2017 for The Data Incubator Fellowship. Linked detailed iPython walk-throughs forthcoming. Please feel free to inquire for more detailed code.

(1) Using Graph and Network Theory To Investigate the NYC Social Elite

<a href="http://www.newyorksocialdiary.com/">The New York Social Diary</a> provides a fascinating lens onto New York's socially well-to-do. As shown in <a href="http://www.newyorksocialdiary.com/party-pictures/2014/holiday-dinners-and-doers
">this report of a recent holiday party</a>, almost all the photos have annotated captions labeling their subjects. We can think of this as implicitly implying a social graph: there is a connection between two individuals if they appear in a picture together.

In this project, I investigate these connections between the NYC elite. I determine who the most socially popular and influential people are, as well as pairs with the strongest connections. (The latter analysis picks up the obvious  -- marriages and family ties -- as well as connections not readily apparent, such as affairs and infidelities.) Methodologically this was completed by scraping all website pages for photo captions, total number of valid captions was 93,093. These captions were then parsed to return unique names, revealing a total of 113,031 names in the social network. Finally, I performed the graph analysis using networkx. The graph visualization can be viewed in the screenshot below.

Tools used: Python - BeutifulSoup (webscrape), regex (cleaning website data), networkx (graph/viz)

[INSERT PHOTO HERE] -- network_graph.png

(2) Using PostGreSQL and Pandas To Investigate NYC Restaurants

The city of New York inspect roughly 24,000 restaraunts a year and assigns a grade to restaurants after each inspection, this creates a public dataset of 531,935 records. I used PostGreSQL and Pandas to parse and analyze four years worth of  NYC Restaurant Inspections data. I extracted different slices -- determining the grade distribution by zipcode, borough, and cuisine. I also found which cuisines tended to have a disproportionate number of which violations.

<center><a href="http://cdb.io/1dkAG2o"><img src="https://github.com/kkamb/miniprojects/blob/master/carto_map.png"></a></center><br>

[INSERT PHOTO HERE] -- carto_map.png
A map view of scores by zipcode, where higher intensity reds are equivalent to higher scores, via <a href="http://cdb.io/1dkAG2o">cartodb</a>.

TOOLS USED: Pandas, numPy, psql, SQLAlchemy, matplotlib, seaborn

(3) Using Semi-Structured Data to Predict Yelp Ratings

I attempted to predict a new venue's popularity from information available when the venue opens. <a href="https://www.yelp.com/developers/documentation/v2/business">The dataset</a> contains unstructured meta data (json) about each venue (city, latitude/longitude, category descriptions, etc), and a star rating. After data munging and feature variable creation, I used Python's Scikit Learn libraries to  create transformations to allow us to model feature data, since the data is unstructured this required json parsing to create flattened nested dictionaries. I then used Scikit Learn to create training and test data to develop several different Machine Learning algorithms.

The predictive algorithms revealed three weak predictors (city, lat/long, attributes) and one fair predictor (categories). I ultimately used FeatureUnion methodology and created and ensemble regressor for the final model. In Python after creating our custom data transformers, the final pipeline looks like this:

'''
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
'''

One pipeline is required for each variable given the unstructured nature of the Yelp data, the transformers fit our data to interact with SciKit Learn. After training and scoring the pipelined model...

'''
pipeline.fit(data_train,stars_train)
print pipeline.predict(data_test[:10])
print stars_test[:10].values
score = pipeline.score(data_test,stars_test)
print score
preds = pipeline.predict(data_test)
print 'Ensemble Model RMSE: ', mean_squared_error(stars_test,preds)**0.5
print 'Ensemble Model MAE: ', mean_absolute_error(stars_test,preds)
print 'Ensemble Model R^2: ', r2_score(stars_test,preds)
'''
----
'''
[ 3.6706147   3.99940255  3.97274505  3.83213406  3.79369298  3.6706147
  3.31230485  3.41025357  3.58801357  3.72347899]
[ 5.   4.   4.5  4.5  3.5  5.   4.   3.5  4.   5. ]
0.0789390332998
Ensemble Model RMSE:  0.850048323084
Ensemble Model MAE:  0.675276799799
Ensemble Model R^2:  0.0789390332998
'''

We have a model that predicts a star rating based on city, lat/lon, category and venue attributes (178 possible). Despite our efforts our model has relatively low predictive power, as indicated by the low R^2 value.

TOOLS USED: numpy, seaborn, pandas, dill, sklearn - KNeighborsRegressor, linear_model, feature_extraction, neighbors, cross_validation, grid_search, train_test_split, numpy, matplotlib

(4) Analyzing Wikipedia via MapReduce

I scraped the entire English Wikipedia to determine its <a href="https://github.com/">most frequently used words</a>. I then collected <a href="https://github.com/">link statistics</a> (mean, standard deviations, percentiles) for the unique links on each page, and <a href="https://github.com/">found the top linked concepts</a> by examining doublelinks.

TOOLS USED: Hadoop/HDFS, MapReduce, MRJob, XML, BeutifulSoup, Custom Builty XML Parser

(5) DataViz

This project required fellows to create a webhosted interactive data visualization tool. My project analyzed historical project data from DonorsChoose.org, a dataset of 950,000 records (~8GB). More information on my project can be <a href="https://github.com/beilmanmich/donors_dashboard">found here</a>.

TOOLS USED: ds.js, html, css, java, Flask, MongoDB, Numpy, Heroku

(6) Using Time Series Machine Learning to Predict the Weather

Time series prediction presents its own challenges which are different from machine-learning problems. As with many other classes of problems, there are a number of common features in these predictions. In this project, I built two time series models to predict weather in a given city at a given time, based on historical trends. Seasonal features are nice because they are relatively safe to extrapolate into the future. There are two ways to handle seasonality.

The simplest (and perhaps most robust) is to have a set of indicator variables. That is, make the assumption that the temperature at any given time is a function of only the month of the year and the hour of the day, and use that to predict the temperature value. A more complex approach is to fit our model to the time series curve. Since we know that temperature is roughly sinusoidal, we know that a resonable model might be:

$$ y_t = y_0 \sin\left(2\pi\frac{t - t_0}{T}\right) + \epsilon $$

where $k$ and $t_0$ are parameters to be learned and $T$ is one year for seasonal variation.  While this is linear in $y_0$, it is not linear in $t_0$. However, we know from Fourier analysis, that the above is
equivalent to

$$ y_t = A \sin\left(2\pi\frac{t}{T}\right) + B \cos\left(2\pi\frac{t}{T}\right) + \epsilon $$

which is linear in $A$ and $B$.

TOOLS USED: Pands, sklearn - BaseEstimator, TransformerMixin,  DictBectorizer, Pipeline, LogisticRegression, KNeighborsRegressor

(7) Using NLP to Predict Yelp Ratings

I then returned to the Yelp Data to explore how much information was contained in the review texts, whether they could more accurately predict ratings. This project allowed a rich dataset to practice natural language processing (NLP) as the unstructured data contains the text record of a written yelp review. Since I was working with over one million reviews and a design matrix of over a million feature-words, scalability was an overriding factor during model selection, especially since the model had to fit within Heroku's memory constraints. However, the predictive power of even a basic out-of-the-box ridge regression was magnitudes greater than that of the models in the previous section (yielding a score of over .6).

TOOLS USED: nltk, nltk.tokenize, nltk.WordNetLemmatizer, Textblob, dill, numpy, pandas, seaborn, matplotlib.pylab, sklearn - linear_models, externals, joblib, cross_validation, grid_searchCV, FeatureUnion, Pipeline, CountVectorizer, HashingVectorizer, Tfidf, Tfidf_vectorizer

(8) Using ML to Categorize Music Samples

Music offers an extremely rich and interesting playing field. The objective of this miniproject is to develop models that are able to recognize the genre of a musical piece (electronic, folkcountry, jazz, raphiphop, rock). first from pre-computed features and then working from the raw waveform (files with 5-10 seconds of a music sample). This is a typical example of a classification problem on time series data. 

TOOLS USED: numpy, scipy, Librosa, audioread, audioop, pandas, dill, sklearn - preprocessing, ensemble, LabelEncoder, DecisionTreeClassifier, StandardScaler, normalize, PCA, RandomForestClassifier, Pipeline, SVM, SVC

(9) SparkOverload - Using Spark to Analyze StackOverflow Data

StackOverflow is a collaboratively edited question-and-answer site originally focused on programming topics. Because of the variety of features tracked, including a variety of feedback metrics, it allows for some open-ended analysis of user behavior on the site.

StackExchange (the parent organization) provides an anonymized <a href="https://archive.org/details/stackexchange">data dump</a>, this project  used Spark to perform data manipulation, analysis, and machine learning on this dataset. Using PySpark and Google Cloud Platform, I was able to perform dsitributed computing on a massive dataset of 9.6GB, unstructured in XML, spanning 341 seperate files. Using Python's Scala API my example workflow is as follows:

(a) Edit source code in your main.py file, classes in a separate classes.py (Class definitions need to be written in a separate file and then included at runtime.)
(b) Run locally on a chunk using eg. `$SPARK_HOME/bin/spark-submit --py-files src/classes.py src/main.py data/stats results/stats/`
(c) Run on GCP once your testing and development are done. 