#!/usr/bin/env python
# coding: utf-8

# In[32]:


get_ipython().system('pip install scikit-surprise')


# In[1]:


import gzip
from collections import defaultdict
import math
import scipy.optimize
from sklearn import svm
from sklearn.preprocessing import LabelEncoder
import numpy as np
import string
import random
import string
from sklearn import linear_model
import pandas as pd
from sklearn.model_selection import train_test_split
import torch
print(torch.__version__)
import torch.nn as nn
import torch.optim as optim
from surprise import Dataset, Reader, SVDpp
from surprise.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split


# In[2]:


import warnings
warnings.filterwarnings("ignore")


# In[3]:


def assertFloat(x):
    assert type(float(x)) == float

def assertFloatList(items, N):
    assert len(items) == N
    assert [type(float(x)) for x in items] == [float]*N


# In[4]:


def readGz(path):
    for l in gzip.open(path, 'rt'):
        yield eval(l)


# In[5]:


def readCSV(path):
    f = gzip.open(path, 'rt')
    f.readline()
    for l in f:
        u,b,r = l.strip().split(',')
        r = int(r)
        yield u,b,r


# In[6]:


# assigment 1 Read prediction: Logistic Regression


# In[7]:


def read_interactions(file_path):
    for user, book, rating in readCSV(file_path):
        yield user, book, int(rating)

# Initialize data structures
data = []
book_set = set()
user_books = defaultdict(set)
book_users = defaultdict(set)
user_id_map = {}
book_id_map = {}

# Read data and build mappings
for user, book, rating in read_interactions("train_Interactions.csv.gz"):
    data.append((user, book, rating))
    book_set.add(book)
    user_books[user].add(book)
    book_users[book].add(user)
    if user not in user_id_map:
        user_id_map[user] = len(user_id_map)
    if book not in book_id_map:
        book_id_map[book] = len(book_id_map)


# In[8]:


# Shuffle data
random.shuffle(data)

# Split data
train_data = data[:160000]
extra_train_data = data[160000:180000]
validation_data = data[180000:]

# Function to get a random book
def get_random_book():
    return random.choice(list(book_set))

# Generate negative samples for validation
validation_samples = {}
for user, book, _ in validation_data:
    validation_samples[(user, book)] = 1  # Positive sample
    # Negative sample
    neg_book = get_random_book()
    while neg_book in user_books[user] or (user, neg_book) in validation_samples:
        neg_book = get_random_book()
    validation_samples[(user, neg_book)] = 0  # Negative sample

# Generate negative samples for extra training
extra_samples = {}
for user, book, _ in extra_train_data:
    extra_samples[(user, book)] = 1  # Positive sample
    # Negative sample
    neg_book = get_random_book()
    while neg_book in user_books[user] or (user, neg_book) in extra_samples:
        neg_book = get_random_book()
    extra_samples[(user, neg_book)] = 0  # Negative sample


# In[9]:


def map_keys_to_indices(pairs):
    user_indices = [user_id_map.get(user, 0) for user, _ in pairs]
    book_indices = [book_id_map.get(book, 0) for _, book in pairs]
    return user_indices, book_indices

def cosine_similarity(set1, set2):
    intersection_size = len(set1 & set2)
    if not set1 or not set2:
        return 0
    similarity = intersection_size / math.sqrt(len(set1) * len(set2))
    return similarity


# In[10]:


class BPRModel(torch.nn.Module):
    def __init__(self, num_users, num_items, latent_dim, reg_bias, reg_latent):
        super(BPRModel, self).__init__()
        self.user_factors = torch.nn.Embedding(num_users, latent_dim)
        self.item_factors = torch.nn.Embedding(num_items, latent_dim)
        self.item_bias = torch.nn.Embedding(num_items, 1)
        self.reg_bias = reg_bias
        self.reg_latent = reg_latent

        # Initialize embeddings
        torch.nn.init.normal_(self.user_factors.weight, std=0.01)
        torch.nn.init.normal_(self.item_factors.weight, std=0.01)
        torch.nn.init.zeros_(self.item_bias.weight)

    def forward(self, user_indices, pos_item_indices, neg_item_indices):
        user_emb = self.user_factors(user_indices)
        pos_item_emb = self.item_factors(pos_item_indices)
        neg_item_emb = self.item_factors(neg_item_indices)
        pos_item_bias = self.item_bias(pos_item_indices).squeeze()
        neg_item_bias = self.item_bias(neg_item_indices).squeeze()

        pos_scores = (user_emb * pos_item_emb).sum(dim=1) + pos_item_bias
        neg_scores = (user_emb * neg_item_emb).sum(dim=1) + neg_item_bias
        loss = -torch.log(torch.sigmoid(pos_scores - neg_scores) + 1e-10).mean()  # Added epsilon for numerical stability

        # Regularization
        reg_loss = self.reg_bias * (pos_item_bias.norm(2) + neg_item_bias.norm(2)) / 2
        reg_loss += self.reg_latent * (user_emb.norm(2).pow(2).mean() +
                                       pos_item_emb.norm(2).pow(2).mean() +
                                       neg_item_emb.norm(2).pow(2).mean())
        return loss + reg_loss

    def predict(self, user_indices, item_indices):
        user_emb = self.user_factors(user_indices)
        item_emb = self.item_factors(item_indices)
        item_bias = self.item_bias(item_indices).squeeze()
        scores = (user_emb * item_emb).sum(dim=1) + item_bias
        return scores


# In[11]:


def train_bpr(model, optimizer, user_item_pairs):
    num_samples = 50000
    user_indices = []
    pos_item_indices = []
    neg_item_indices = []
    for _ in range(num_samples):
        user, pos_item, _ = random.choice(user_item_pairs)
        neg_item = get_random_book()
        while neg_item in user_books[user]:
            neg_item = get_random_book()
        user_indices.append(user_id_map[user])
        pos_item_indices.append(book_id_map[pos_item])
        neg_item_indices.append(book_id_map[neg_item])
    user_indices = torch.tensor(user_indices, dtype=torch.long)
    pos_item_indices = torch.tensor(pos_item_indices, dtype=torch.long)
    neg_item_indices = torch.tensor(neg_item_indices, dtype=torch.long)

    # Validate indices
    assertFloatList(user_indices.tolist(), num_samples)
    assertFloatList(pos_item_indices.tolist(), num_samples)
    assertFloatList(neg_item_indices.tolist(), num_samples)

    model.train()
    optimizer.zero_grad()
    loss = model(user_indices, pos_item_indices, neg_item_indices)
    loss.backward()
    optimizer.step()
    return loss.item()


# In[13]:


num_users = len(user_id_map)
num_items = len(book_id_map)
latent_dim = 10
reg_bias = 0.00013
reg_latent = 0.00018
bpr_model = BPRModel(num_users, num_items, latent_dim, reg_bias, reg_latent)
optimizer = torch.optim.Adam(bpr_model.parameters(), lr=0.03)

# Prepare validation data
val_user_indices, val_item_indices = map_keys_to_indices(validation_samples.keys())
val_user_tensor = torch.tensor(val_user_indices, dtype=torch.long)
val_item_tensor = torch.tensor(val_item_indices, dtype=torch.long)

for epoch in range(51):
    loss = train_bpr(bpr_model, optimizer, train_data)
    if epoch % 10 == 0:
        print(f"Epoch {epoch}, Loss: {loss:.4f}")
        bpr_model.eval()
        with torch.no_grad():
            scores = bpr_model.predict(val_user_tensor, val_item_tensor).numpy()
        predictions = {pair: int(score > 0) for pair, score in zip(validation_samples.keys(), scores)}
        accuracy = sum(predictions[pair] == label for pair, label in validation_samples.items()) / len(validation_samples)
        print(f"Validation Accuracy: {accuracy:.4f}")


# In[14]:


# Build book popularity
book_popularity = defaultdict(int)
for _, book, _ in data:
    book_popularity[book] += 1

def compute_features(pairs):
    features = []
    for user, book in pairs:
        user_books_set = user_books.get(user, set())
        item_similarity = [cosine_similarity(book_users.get(other_book, set()), book_users.get(book, set()))
                           for other_book in user_books_set if other_book != book]
        avg_item_sim = sum(item_similarity) / len(item_similarity) if item_similarity else 0.01
        max_item_sim = max(item_similarity) if item_similarity else 0.01

        book_users_set = book_users.get(book, set())
        user_similarity = [cosine_similarity(user_books.get(other_user, set()), user_books_set)
                           for other_user in book_users_set if other_user != user]
        avg_user_sim = sum(user_similarity) / len(user_similarity) if user_similarity else 0.01
        max_user_sim = max(user_similarity) if user_similarity else 0.01

        # BPR score
        user_idx = torch.tensor([user_id_map.get(user, 0)], dtype=torch.long)
        book_idx = torch.tensor([book_id_map.get(book, 0)], dtype=torch.long)
        with torch.no_grad():
            bpr_score = bpr_model.predict(user_idx, book_idx).item()

        # Validate BPR score
        assertFloat(bpr_score)

        # Book popularity
        popularity = book_popularity.get(book, 0) / 100

        # Feature vector
        feature_vector = [bpr_score, avg_item_sim, max_item_sim, avg_user_sim, max_user_sim, popularity, 1]
        # Validate feature vector
        assertFloatList(feature_vector, 7)
        features.append(feature_vector)
    return features

# Prepare training data for logistic regression
X_train = compute_features(extra_samples.keys())
y_train = list(extra_samples.values())

# Validate labels
assert all(label in [0, 1] for label in y_train)

# Train logistic regression model
log_reg = linear_model.LogisticRegression(fit_intercept=False, max_iter=1000)
log_reg.fit(X_train, y_train)


# In[15]:


X_val = compute_features(validation_samples.keys())
y_val = list(validation_samples.values())
y_pred = log_reg.predict(X_val)
accuracy = sum(yp == yt for yp, yt in zip(y_pred, y_val)) / len(y_val)
print(f"Validation Accuracy after Logistic Regression: {accuracy:.4f}")


# In[16]:


# Read test pairs
test_pairs = []
with open("pairs_Read.csv", "r") as f:
    next(f)  # Skip header
    for line in f:
        user, book = line.strip().split(',')
        test_pairs.append((user, book))


# In[17]:


# Compute features for test data
X_test = compute_features(test_pairs)

# Validate features
for feature_vector in X_test:
    assertFloatList(feature_vector, 7)

# Predict scores
test_scores = log_reg.decision_function(X_test)


# In[18]:


# Apply thresholding per user
def threshold_predictions(pairs, scores):
    predictions = {}
    user_to_scores = defaultdict(list)
    for (user, book), score in zip(pairs, scores):
        user_to_scores[user].append((book, score))
    for user, items in user_to_scores.items():
        items.sort(key=lambda x: x[1], reverse=True)
        threshold_score = items[len(items) // 2][1]
        for book, score in items:
            predictions[(user, book)] = int(score > threshold_score)
    return predictions

final_predictions = threshold_predictions(test_pairs, test_scores)

# Write predictions to file
with open("predictions_Read.csv", 'w') as f:
    f.write('userID,bookID,prediction\n')
    for user, book in test_pairs:
        pred = final_predictions.get((user, book), 0)
        f.write(f"{user},{book},{pred}\n")


# In[19]:


# Rating prediction:


# In[20]:


# Load training data
train_data = []
for user, book, rating in readCSV('train_Interactions.csv.gz'):
    train_data.append((user, book, rating))

# Convert to DataFrame
df_train = pd.DataFrame(train_data, columns=['userID', 'bookID', 'rating'])

# Optional: Split data into training and validation sets
train_df, val_df = train_test_split(df_train, test_size=0.1, random_state=42)

# Define the rating scale
rating_scale = (df_train['rating'].min(), df_train['rating'].max())

# Create a Reader
reader = Reader(rating_scale=rating_scale)

# Load data into Surprise dataset
data = Dataset.load_from_df(train_df[['userID', 'bookID', 'rating']], reader)


# In[28]:


param_grid = {
    'n_factors': [5, 10, 15, 20, 25],
    'lr_all': [0.0005, 0.001, 0.002, 0.003],
    'reg_all': [0.1, 0.15, 0.2, 0.3, 0.5],
    'n_epochs': [75, 100, 125, 150]
}


# In[29]:


# Perform grid search
gs = GridSearchCV(SVDpp, param_grid, measures=['rmse'], cv=3, n_jobs=-1)
gs.fit(data)

# Output best score and parameters
print(f"Best RMSE: {gs.best_score['rmse']}")
print(f"Best params: {gs.best_params['rmse']}")


# In[30]:


# Get the best parameters
best_params = gs.best_params['rmse']

# Build full training set
trainset = data.build_full_trainset()

# Train the SVDpp model with best parameters
model = SVDpp(n_factors=best_params['n_factors'],
              lr_all=best_params['lr_all'],
              reg_all=best_params['reg_all'],
              n_epochs=best_params['n_epochs'])
model.fit(trainset)

# Read test pairs
test_pairs = []
with open('pairs_Rating.csv', 'r') as f:
    next(f)  # Skip header
    for line in f:
        user, book = line.strip().split(',')
        test_pairs.append((user, book))

# Make predictions
with open('predictions_Rating.csv', 'w') as pred_file:
    pred_file.write('userID,bookID,prediction\n')
    for user, book in test_pairs:
        est = model.predict(user, book).est
        pred_file.write(f"{user},{book},{est}\n")


# In[ ]:


#-----------------------------------------------------------------------------------------


# In[ ]:


# Bi-encoder
torch.manual_seed(42)
np.random.seed(42)
random.seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)

# Determine the device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Function to read the CSV file
def readCSV(path):
    f = gzip.open(path, 'rt')
    f.readline()  # Skip the header
    for l in f:
        u, b, r = l.strip().split(',')
        r = int(r)
        yield u, b, r

# Load all ratings from training data
allRatings = []
for l in readCSV("train_Interactions.csv.gz"):
    allRatings.append(l)

# Convert to DataFrame for easier manipulation
ratings_df = pd.DataFrame(allRatings, columns=['userID', 'bookID', 'rating'])

# Load test data
test_df = pd.read_csv("pairs_Read.csv")

# Combine users and books from both training and test data
all_users = set(ratings_df['userID']).union(set(test_df['userID']))
all_books = set(ratings_df['bookID']).union(set(test_df['bookID']))

# Create dictionaries for quick lookup
books_per_user = defaultdict(set)
users_per_book = defaultdict(set)
for _, row in ratings_df.iterrows():
    u = row['userID']
    b = row['bookID']
    books_per_user[u].add(b)
    users_per_book[b].add(u)

# Negative sampling
negative_ratio = 1  # Adjust as needed
negative_samples = []
for u in books_per_user:
    read_books = books_per_user[u]
    unread_books = all_books - read_books
    n_negative = min(len(read_books) * negative_ratio, len(unread_books))
    if n_negative > 0:
        # Convert unread_books to a list before sampling
        negative_books = random.sample(list(unread_books), n_negative)
        for b in negative_books:
            negative_samples.append((u, b, 0))  # Label 0 for unread books

# Positive samples (label 1)
positive_samples = [(row['userID'], row['bookID'], 1) for _, row in ratings_df.iterrows()]

# Combine samples
all_samples = positive_samples + negative_samples
samples_df = pd.DataFrame(all_samples, columns=['userID', 'bookID', 'label'])

# Shuffle the data
samples_df = samples_df.sample(frac=1, random_state=42).reset_index(drop=True)

# Create ID to index mappings
user_ids = list(all_users)
book_ids = list(all_books)

user_to_index = {u: idx for idx, u in enumerate(user_ids)}
book_to_index = {b: idx for idx, b in enumerate(book_ids)}

num_users = len(user_ids)
num_books = len(book_ids)

# Prepare input data
def prepare_input(df):
    user_indices = df['userID'].map(user_to_index).astype(int).values
    book_indices = df['bookID'].map(book_to_index).astype(int).values
    labels = df['label'].values
    return user_indices, book_indices, labels

# Split into training and validation sets
train_df, val_df = train_test_split(samples_df, test_size=0.2, random_state=42, stratify=samples_df['label'])

train_user_indices, train_book_indices, train_labels = prepare_input(train_df)
val_user_indices, val_book_indices, val_labels = prepare_input(val_df)

# Define the model
class BiEncoderModel(nn.Module):
    def __init__(self, num_users, num_books, embedding_dim):
        super(BiEncoderModel, self).__init__()
        self.user_embedding = nn.Embedding(num_users, embedding_dim)
        self.book_embedding = nn.Embedding(num_books, embedding_dim)
        
        # Initialize embeddings
        nn.init.xavier_uniform_(self.user_embedding.weight)
        nn.init.xavier_uniform_(self.book_embedding.weight)
        
    def forward(self, user_indices, book_indices):
        user_embeds = self.user_embedding(user_indices)
        book_embeds = self.book_embedding(book_indices)
        scores = (user_embeds * book_embeds).sum(dim=1)
        probs = torch.sigmoid(scores)
        return probs

embedding_dim = 50  # Adjust as needed
model = BiEncoderModel(num_users, num_books, embedding_dim).to(device)

# Loss function and optimizer
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Prepare DataLoaders
from torch.utils.data import Dataset, DataLoader

class InteractionDataset(Dataset):
    def __init__(self, user_indices, book_indices, labels):
        self.user_indices = torch.LongTensor(user_indices)
        self.book_indices = torch.LongTensor(book_indices)
        self.labels = torch.FloatTensor(labels)
        
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        return self.user_indices[idx], self.book_indices[idx], self.labels[idx]

# Create datasets
train_dataset = InteractionDataset(train_user_indices, train_book_indices, train_labels)
val_dataset = InteractionDataset(val_user_indices, val_book_indices, val_labels)

# Create DataLoaders
batch_size = 1024  # Adjust as needed
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size)

# Training loop
num_epochs = 5  # Adjust as needed

for epoch in range(num_epochs):
    model.train()
    total_loss = 0
    for user_indices, book_indices, labels in train_loader:
        user_indices = user_indices.to(device)
        book_indices = book_indices.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = model(user_indices, book_indices)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * labels.size(0)
    avg_loss = total_loss / len(train_dataset)
    
    # Validation
    model.eval()
    total_val_loss = 0
    correct = 0
    with torch.no_grad():
        for user_indices, book_indices, labels in val_loader:
            user_indices = user_indices.to(device)
            book_indices = book_indices.to(device)
            labels = labels.to(device)

            outputs = model(user_indices, book_indices)
            loss = criterion(outputs, labels)
            total_val_loss += loss.item() * labels.size(0)
            preds = (outputs >= 0.5).float()
            correct += (preds == labels).sum().item()
    avg_val_loss = total_val_loss / len(val_dataset)
    accuracy = correct / len(val_dataset)
    
    print(f"Epoch {epoch+1}/{num_epochs}, Loss: {avg_loss:.4f}, Val Loss: {avg_val_loss:.4f}, Val Acc: {accuracy:.4f}")

# Prepare test input
def prepare_test_input(df):
    user_indices = df['userID'].map(user_to_index).astype(int).values
    book_indices = df['bookID'].map(book_to_index).astype(int).values
    return torch.LongTensor(user_indices), torch.LongTensor(book_indices)

test_user_indices, test_book_indices = prepare_test_input(test_df)

# Create test DataLoader
test_dataset = torch.utils.data.TensorDataset(test_user_indices, test_book_indices)
test_loader = DataLoader(test_dataset, batch_size=1024)

# Make predictions
model.eval()
all_predictions = []
with torch.no_grad():
    for user_indices, book_indices in test_loader:
        user_indices = user_indices.to(device)
        book_indices = book_indices.to(device)
        outputs = model(user_indices, book_indices)
        all_predictions.extend(outputs.cpu().tolist())

# Convert probabilities to binary labels
threshold = 0.5
binary_predictions = [1 if pred >= threshold else 0 for pred in all_predictions]

# Write predictions to CSV
test_df['prediction'] = binary_predictions
test_df[['userID', 'bookID', 'prediction']].to_csv('predictions_Read.csv', index=False)


# In[29]:


# Homework 3 code below


# In[16]:


answers = {}


# In[17]:


# Some data structures that will be useful


# In[18]:


allRatings = []
for l in readCSV("train_Interactions.csv.gz"):
    allRatings.append(l)


# In[19]:


len(allRatings)


# In[20]:


df = pd.read_csv("pairs_Read.csv")
print(df.head(10))


# In[21]:


allRatings[:10]


# In[22]:


ratingsTrain = allRatings[:190000]
ratingsValid = allRatings[190000:]
ratingsPerUser = defaultdict(list)
ratingsPerItem = defaultdict(list)
for u,b,r in ratingsTrain:
    ratingsPerUser[u].append((b,r))
    ratingsPerItem[b].append((u,r))


# In[23]:


##################################################
# Read prediction                                #
##################################################


# In[24]:


# Copied from baseline code
bookCount = defaultdict(int)
totalRead = 0

for user,book,_ in readCSV("train_Interactions.csv.gz"):
    bookCount[book] += 1
    totalRead += 1

mostPopular = [(bookCount[x], x) for x in bookCount]
mostPopular.sort()
mostPopular.reverse()

return1 = set()
count = 0
for ic, i in mostPopular:
    count += ic
    return1.add(i)
    if count > totalRead/2: break


# In[25]:


### Question 1


# In[26]:


# Find all books in the dataset and create the negative valid set
allBooks = set([b for _,b,_ in allRatings])

ratingsValidNeg = []
for u,b,r in ratingsValid:
    bookReadByUser = set([b for b,_ in ratingsPerUser[u]])
    unread = list(allBooks - bookReadByUser)
    
    if unread:
        negativeB = random.choice(unread)
        ratingsValidNeg.append((u, negativeB, 0))


# In[27]:


combinedValid = ratingsValid + ratingsValidNeg
def predict(user, book, mostPopularBooks):
    return 1 if book in mostPopularBooks else 0

correct = 0
total = 0
for u,b,r in combinedValid:
    prediction = predict(u,b,return1)
    binary = 0
    if r > 0:
        binary = 1
    if prediction == binary:
        correct += 1
    total += 1
acc1 = correct/total
acc1


# In[28]:


answers['Q1'] = acc1


# In[19]:


assertFloat(answers['Q1'])
answers


# In[20]:


### Question 2


# In[21]:


def most_accurate_threshold(percentage):
    popular_books = set()
    count = 0
    for ic, i in mostPopular:
        count += ic
        popular_books.add(i)
        if count > totalRead * percentage: break
    return popular_books

perc = [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]
best_threshold = 0
best_acc = 0

for p in perc:
    return1 = most_accurate_threshold(p)
    
    correct = 0
    total = 0
    for u,b,r in combinedValid:
        prediction = predict(u,b,return1)
        binary = 0
        if r > 0:
            binary = 1
        if prediction == binary:
            correct += 1
        total += 1
    acc = correct/total
    
    if acc > best_acc:
        best_acc = acc
        best_threshold = p


threshold = best_threshold
acc2 = best_acc


# In[22]:


answers['Q2'] = [threshold, acc2]
answers


# In[23]:


assertFloat(answers['Q2'][0])
assertFloat(answers['Q2'][1])


# In[24]:


### Question 3/4


# In[25]:


def Jaccard(s1, s2):
    numer = len(s1.intersection(s2))
    denom = len(s1.union(s2))
    if denom == 0:
        return 0
    return numer / denom


# In[26]:


usersPerBook = defaultdict(set)
booksPerUser = defaultdict(set)

# Create a dictionary that maps each book to its set of readers
for u,b,_ in ratingsTrain:
    usersPerBook[b].add(u)
    booksPerUser[u].add(b)

# for u,b,_ in allRatings:
#     usersPerBook[b].add(u)
    
# Prepare the validation set
validation_set = [(u,b,1) for u,b,_ in ratingsValid] + ratingsValidNeg


# In[27]:


threshold = 0.004
predictions = list()
labels = list()
for user, book, label in validation_set:
    labels.append(label)
    similarities = [0]
    user_b = {user for user in usersPerBook[book]}
    for book_prime in booksPerUser[user]:
        user_b_prime = {user for user in usersPerBook[book_prime]}
        sim = Jaccard(user_b, user_b_prime)
        similarities.append(sim)
    if max(similarities) > threshold:
        predictions.append(1)
    else:
        predictions.append(0)


# In[28]:


corrects = [p == l for p, l in zip(predictions, labels)]
acc3 = sum(corrects) / len(predictions)
acc3


# In[29]:


# maxJaccard = {}
# for u, b, _ in validation_set:
#     maxSim = 0
#     if b in usersPerBook:
#         for b_prime in booksPerUser[u]:
#             if b_prime in usersPerBook:
#                 sim = Jaccard(usersPerBook[b], usersPerBook[b_prime])
#                 if sim > maxSim:
#                     maxSim = sim
#     maxJaccard[(u, b)] = maxSim
# # maxJaccard


# In[30]:


# thresholds = [i / 10000 for i in range(0, 200)]  # 0.000 to 0.020 in steps of 0.001

# # Store accuracy for each threshold
# accuracy_per_threshold = []

# for threshold in thresholds:
#     correct_predictions = 0
#     total_predictions = len(validation_set)

#     for u, b, actual_label in validation_set:
#         predicted_label = 1 if maxJaccard[(u, b)] >= threshold else 0
#         if predicted_label == actual_label:
#             correct_predictions += 1
#     accuracy = correct_predictions / total_predictions

#     accuracy_per_threshold.append({
#         'threshold': threshold,
#         'accuracy': accuracy
#     })

# best_performance = max(accuracy_per_threshold, key=lambda x: x['accuracy'])
# best_threshold = best_performance['threshold']

# print(f"Best Threshold: {best_threshold}")
# print(f"Accuracy: {best_performance['accuracy']:.4f}")


# In[31]:


threshold_Jaccard = 0.004
threshold_popularity = 24
predictions = list()
labels = list()
popularityL = list()
for user, book, label in validation_set:
    labels.append(label)
    similarities = [0]
    user_b = {user for user in usersPerBook[book]}
    popularity = len(user_b)
    popularityL.append(popularity)
    for book_prime in booksPerUser[user]:
        user_b_prime = {user for user in usersPerBook[book_prime]}
        sim = Jaccard(user_b, user_b_prime)
        similarities.append(sim)
    if max(similarities) > threshold_Jaccard and popularity > threshold_popularity:
        predictions.append(1)
    else:
        predictions.append(0)

corrects = [p == l for p, l in zip(predictions, labels)]
acc4 = sum(corrects) / len(predictions)
acc4


# In[32]:


sorted(popularityL)[10000]


# In[33]:


answers['Q3'] = acc3
answers['Q4'] = acc4


# In[34]:


assertFloat(answers['Q3'])
assertFloat(answers['Q4'])


# In[35]:


threshold_Jaccard = 0.004
threshold_popularity = 24
popularityL = list()

predictions = open("predictions_Read.csv", 'w')
for l in open("pairs_Read.csv"):
    if l.startswith("userID"):
        predictions.write(l)
        continue
    u,b = l.strip().split(',')
    # My code starts here

    similarities = [0]
    user_b = {user for user in usersPerBook[b]}
    popularity = len(user_b)
    popularityL.append(popularity)
    for book_prime in booksPerUser[u]:
        user_b_prime = {user for user in usersPerBook[book_prime]}
        sim = Jaccard(user_b, user_b_prime)
        similarities.append(sim)
        
    if max(similarities) > threshold_Jaccard and popularity > threshold_popularity:
        pred = 1  
    else:
        pred = 0  # Predict 'not read'

    # Write the prediction to the output file
    predictions.write(f"{u},{b},{pred}\n")
    

predictions.close()


# In[36]:


len(popularityL)
sorted(popularityL)[10000]


# In[37]:


answers['Q5'] = "I confirm that I have uploaded an assignment submission to gradescope"


# In[38]:


assert type(answers['Q5']) == str


# In[39]:


##################################################
# Rating prediction                              #
##################################################


# In[40]:


ratingsValid[:10]


# In[41]:


### Question 6


# In[42]:


# Dataframe preprocessing
df_train = pd.DataFrame(ratingsTrain, columns=['userID', 'itemID', 'rating'])
df_valid = pd.DataFrame(ratingsValid, columns=['userID', 'itemID', 'rating'])

df_valid = df_valid[
    df_valid['userID'].isin(df_train['userID']) &
    df_valid['itemID'].isin(df_train['itemID'])
].copy()

user_encoder = LabelEncoder()
item_encoder = LabelEncoder()

# Fit encoders on training data
df_train['user_idx'] = user_encoder.fit_transform(df_train['userID'])
df_train['item_idx'] = item_encoder.fit_transform(df_train['itemID'])

# Transform validation data
df_valid['user_idx'] = user_encoder.transform(df_valid['userID'])
df_valid['item_idx'] = item_encoder.transform(df_valid['itemID'])


# In[43]:


alpha = df_train['rating'].mean()
alpha


# In[44]:


# Initialize bias term for every user and item
num_users = df_train['user_idx'].nunique()
num_items = df_train['item_idx'].nunique()
beta_user = np.zeros(num_users)
beta_item = np.zeros(num_items)


# In[45]:


learning_rate = 0.01
num_epochs = 10
lambda_reg = 1  # reg parameter

train_users = df_train['user_idx'].values
train_items = df_train['item_idx'].values
train_ratings = df_train['rating'].values

for epoch in range(num_epochs):
    shuffled_indices = np.arange(len(train_ratings))
    np.random.shuffle(shuffled_indices)
    
    for idx in shuffled_indices:
        u = train_users[idx]
        i = train_items[idx]
        r_ui = train_ratings[idx]
        
        # make rating predictions
        pred = alpha + beta_user[u] + beta_item[i]
        
        # Compute the error
        error = r_ui - pred
        
        # Update biases with regularization
        beta_user[u] += learning_rate * (error - lambda_reg * beta_user[u])
        beta_item[i] += learning_rate * (error - lambda_reg * beta_item[i])
    
    # Optionally, compute training error to monitor convergence
    train_preds = alpha + beta_user[train_users] + beta_item[train_items]
    train_mse = np.mean((train_preds - train_ratings) ** 2)
    print(f"Epoch {epoch+1}/{num_epochs}, Training MSE: {train_mse:.4f}")


# In[46]:


valid_users = df_valid['user_idx'].values
valid_items = df_valid['item_idx'].values
valid_ratings = df_valid['rating'].values

# Predict ratings
valid_preds = alpha + beta_user[valid_users] + beta_item[valid_items]
valid_preds[:10]


# In[47]:


from sklearn.metrics import mean_squared_error

mse = mean_squared_error(valid_ratings, valid_preds)
print(f"Validation MSE: {mse:.4f}")
validMSE = mse


# In[48]:


answers['Q6'] = validMSE
answers


# In[49]:


assertFloat(answers['Q6'])


# In[50]:


### Question 7


# In[51]:


maxBeta = float(max(beta_user))
minBeta = float(min(beta_user))
#
max_beta_index = np.argmax(beta_user)
max_user_id = user_encoder.inverse_transform([max_beta_index])[0]
maxUser = max_user_id

min_beta_index = np.argmin(beta_user)
min_user_id = user_encoder.inverse_transform([min_beta_index])[0]
minUser = min_user_id
minUser


# In[52]:


answers['Q7'] = [maxUser, minUser, maxBeta, minBeta]


# In[53]:


assert [type(x) for x in answers['Q7']] == [str, str, float, float]


# In[54]:


### Question 8


# In[55]:


learning_rate = 0.01
num_epochs = 10
lambda_reg = 0.5  # reg parameter

train_users = df_train['user_idx'].values
train_items = df_train['item_idx'].values
train_ratings = df_train['rating'].values

for epoch in range(num_epochs):
    shuffled_indices = np.arange(len(train_ratings))
    np.random.shuffle(shuffled_indices)
    
    for idx in shuffled_indices:
        u = train_users[idx]
        i = train_items[idx]
        r_ui = train_ratings[idx]
        
        # make rating predictions
        pred = alpha + beta_user[u] + beta_item[i]
        
        # Compute the error
        error = r_ui - pred
        
        # Update biases with regularization
        beta_user[u] += learning_rate * (error - lambda_reg * beta_user[u])
        beta_item[i] += learning_rate * (error - lambda_reg * beta_item[i])
    
    # Optionally, compute training error to monitor convergence
    train_preds = alpha + beta_user[train_users] + beta_item[train_items]
    train_mse = np.mean((train_preds - train_ratings) ** 2)
    print(f"Epoch {epoch+1}/{num_epochs}, Training MSE: {train_mse:.4f}")
    
valid_preds = alpha + beta_user[valid_users] + beta_item[valid_items]
mse = mean_squared_error(valid_ratings, valid_preds)
print(f"Validation MSE: {mse:.4f}")
validMSE = mse
lamb = lambda_reg


# In[56]:


answers['Q8'] = (lamb, validMSE)


# In[57]:


assertFloat(answers['Q8'][0])
assertFloat(answers['Q8'][1])


# In[60]:


# Create mappings from userID/itemID to indices
userID_to_idx = {label: idx for idx, label in enumerate(user_encoder.classes_)}
itemID_to_idx = {label: idx for idx, label in enumerate(item_encoder.classes_)}

predictions = open("predictions_Rating.csv", 'w')
for l in open("pairs_Rating.csv"):
    if l.startswith("userID"): # header
        predictions.write(l)
        continue
    u,b = l.strip().split(',') # Read the user and item from the "pairs" file and write out your prediction
    
    # Map userID and itemID to indices
    if u in userID_to_idx:
        u_idx = userID_to_idx[u]
        beta_u = beta_user[u_idx]
    else:
        # Handle unseen user
        beta_u = 0.0  # or use np.mean(beta_user)
    
    if b in itemID_to_idx:
        b_idx = itemID_to_idx[b]
        beta_i = beta_item[b_idx]
    else:
        # Handle unseen item
        beta_i = 0.0  # or use np.mean(beta_item)
    
    # Compute the prediction
    pred = alpha + beta_u + beta_i
    
    # Clip the prediction to the valid rating range (e.g., 1 to 5)
    pred = min(max(pred, 1), 5)
    
    # Write out your prediction
    predictions.write(f"{u},{b},{pred}\n")
    
predictions.close()


# In[61]:


f = open("answers_hw3.txt", 'w')
f.write(str(answers) + '\n')
f.close()


# In[ ]:




