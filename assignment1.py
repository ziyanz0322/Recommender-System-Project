import gzip
from collections import defaultdict
import string
import re
from sklearn import linear_model
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.stem import PorterStemmer
import nltk

# Download NLTK data
try:
    nltk.data.find("tokenizers/punkt")
except LookupError:
    nltk.download("punkt", quiet=True)

stemmer = PorterStemmer()


def readGz(path):
    for l in gzip.open(path, "rt"):
        yield eval(l)


def readCSV(path):
    if path.endswith(".gz"):
        f = gzip.open(path, "rt")
    else:
        f = open(path, "rt")
    f.readline()
    for l in f:
        parts = l.strip().split(",")
        if len(parts) == 3:
            yield parts[0], parts[1], parts[2]
        else:
            yield parts[0], parts[1]
    f.close()


def Jaccard(s1, s2):
    numer = len(s1.intersection(s2))
    denom = len(s1.union(s2))
    return numer / denom if denom > 0 else 0


###############################################################################
# TASK 1: RATING PREDICTION
###############################################################################


def getGlobalAverage(trainRatings):
    total = sum([r for u, b, r in trainRatings])
    return total / len(trainRatings)


def alphaUpdate(ratingsTrain, alpha, betaU, betaI, lamb):
    total = 0
    for u, b, r in ratingsTrain:
        bu = betaU.get(u, 0)
        bi = betaI.get(b, 0)
        total += r - bu - bi
    return total / len(ratingsTrain)


def betaUUpdate(ratingsPerUser, alpha, betaU, betaI, lamb):
    newBetaU = {}
    for u in ratingsPerUser:
        total = 0
        for b, r in ratingsPerUser[u]:
            bi = betaI.get(b, 0)
            total += r - alpha - bi
        newBetaU[u] = total / (lamb + len(ratingsPerUser[u]))
    return newBetaU


def betaIUpdate(ratingsPerItem, alpha, betaU, betaI, lamb):
    newBetaI = {}
    for b in ratingsPerItem:
        total = 0
        for u, r in ratingsPerItem[b]:
            bu = betaU.get(u, 0)
            total += r - alpha - bu
        newBetaI[b] = total / (lamb + len(ratingsPerItem[b]))
    return newBetaI


def trainModel(ratingsTrain, ratingsPerUser, ratingsPerItem, lamb, iterations):
    alpha = getGlobalAverage(ratingsTrain)
    betaU = {}
    betaI = {}
    for iteration in range(iterations):
        alpha = alphaUpdate(ratingsTrain, alpha, betaU, betaI, lamb)
        betaU = betaUUpdate(ratingsPerUser, alpha, betaU, betaI, lamb)
        betaI = betaIUpdate(ratingsPerItem, alpha, betaU, betaI, lamb)
    return alpha, betaU, betaI


# Load ratings
ratingsTrain = []
for user, book, rating in readCSV("train_Interactions.csv.gz"):
    ratingsTrain.append((user, book, float(rating)))

# Build data structures
ratingsPerUser = defaultdict(list)
ratingsPerItem = defaultdict(list)
for u, b, r in ratingsTrain:
    ratingsPerUser[u].append((b, r))
    ratingsPerItem[b].append((u, r))

# Train with optimized parameters
alpha, betaU, betaI = trainModel(ratingsTrain, ratingsPerUser, ratingsPerItem, 5.0, 100)

# Generate predictions
predictions = open("predictions_Rating.csv", "w")
for l in open("pairs_Rating.csv"):
    if l.startswith("userID"):
        predictions.write(l)
        continue
    u, b = l.strip().split(",")
    bu = betaU.get(u, 0)
    bi = betaI.get(b, 0)
    pred = max(1, min(5, alpha + bu + bi))
    predictions.write(f"{u},{b},{pred}\n")
predictions.close()

###############################################################################
# TASK 2: READ PREDICTION
###############################################################################


def jaccardThresh(u, b, ratingsPerItem, ratingsPerUser):
    usersB = set([user for user, _ in ratingsPerItem.get(b, [])])
    itemsU = set([item for item, _ in ratingsPerUser.get(u, [])])
    maxSim = 0
    for otherItem in itemsU:
        usersOther = set([user for user, _ in ratingsPerItem.get(otherItem, [])])
        sim = Jaccard(usersB, usersOther)
        maxSim = max(maxSim, sim)
    if maxSim > 0.03 or len(ratingsPerItem.get(b, [])) > 25:
        return 1
    return 0


predictions = open("predictions_Read.csv", "w")
for l in open("pairs_Read.csv"):
    if l.startswith("userID"):
        predictions.write(l)
        continue
    u, b = l.strip().split(",")
    pred = jaccardThresh(u, b, ratingsPerItem, ratingsPerUser)
    predictions.write(f"{u},{b},{pred}\n")
predictions.close()

###############################################################################
# TASK 3: CATEGORY PREDICTION
###############################################################################


def preprocess_text(text):
    if not text:
        return ""
    text = text.lower()
    text = re.sub(r"[{}]+".format(re.escape(string.punctuation)), " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def custom_tokenizer(text):
    text = preprocess_text(text)
    words = text.split()
    return [stemmer.stem(word) for word in words]


dataTrain = []
for d in readGz("train_Category.json.gz"):
    dataTrain.append(d)

y_train = [d["genreID"] for d in dataTrain]

# Preprocess texts
train_texts = [preprocess_text(d.get("review_text", "")) for d in dataTrain]

# Apply TF-IDF vectorizer with optimizations
vectorizer = TfidfVectorizer(
    max_features=3000,
    min_df=2,
    max_df=0.85,
    tokenizer=custom_tokenizer,
    lowercase=False,
    stop_words="english",
    ngram_range=(1, 2),
    sublinear_tf=True,
)

X_train = vectorizer.fit_transform(train_texts)

# Train logistic regression model
model = linear_model.LogisticRegression(
    C=0.5, max_iter=3000, multi_class="multinomial", solver="lbfgs", random_state=42
)
model.fit(X_train, y_train)

# Process test data
dataTest = []
for d in readGz("test_Category.json.gz"):
    dataTest.append(d)

test_texts = [preprocess_text(d.get("review_text", "")) for d in dataTest]
X_test = vectorizer.transform(test_texts)
pred_test = model.predict(X_test)

# Write predictions
predictions = open("predictions_Category.csv", "w")
pos = 0
for l in open("pairs_Category.csv"):
    if l.startswith("userID"):
        predictions.write(l)
        continue
    u, b = l.strip().split(",")
    predictions.write(f"{u},{b},{pred_test[pos]}\n")
    pos += 1
predictions.close()
