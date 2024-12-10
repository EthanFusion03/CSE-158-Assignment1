## CSE-158-Assignment1

### Approaches
1. Read Prediction Task:

I used a BPR (Bayesian Personalized Ranking) model to learn user-item preferences, generating scores for user-book pairs. These scores, combined with similarity and popularity features, were input into a logistic regression model to classify each pair as read or unread.

2. Rating Prediction Task:

An SVD++ matrix factorization model was used to predict ratings by learning latent factors for users and items. Hyperparameters were tuned via grid search to minimize Mean Squared Error (MSE) and optimize rating predictions.

### Results
| Metric                  | Public Score | Private Score |
|-------------------------|--------------|---------------|
| Rating Prediction       | 1.4146       | 1.4591        |
| Read Prediction         | 0.7932       | 0.7920        |


![Screenshot 2024-12-09 193625](https://github.com/user-attachments/assets/355bbc13-1c13-4497-ae1f-7cf4b1a83133)
