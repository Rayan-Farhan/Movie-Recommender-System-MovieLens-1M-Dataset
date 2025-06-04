# Movie-Recommender-System-MovieLens-1M-Dataset

This project demonstrates various techniques for building a movie recommender system using user rating data. Techniques like **User-based CF**, **Item-based CF**, **Matrix Factorization**, and **Neural Networks Autoencoders**.Models are evaluated using metrics such as **MAE** and **RMSE**.

---

## Dataset

The system uses two CSV files:

- `movies.csv`: Contains `MovieID`, `Title`, and `Genres`.
- `ratings.csv`: Contains `UserID`, `MovieID`, and `Rating`.

---

## Exploratory Data Analysis

The EDA phase includes:

- Analyzing **most frequent genres**.
- Plotting **movie releases per year**.
- Showing **distribution of ratings**.
- Calculating **average rating per genre**.

### Visuals

- Bar plot of top 15 genres.
- Line chart of number of movies released per year.
- Rating distribution count plot.
- Average rating per genre bar plot.

---

## Models

I implemented:

## 1. **Item-Based Collaborative Filtering**

- Computes cosine similarity between **items** (movies).
- Returns top-N similar movies for a given movie.
- Example: Top 5 recommendations for *Toy Story (1995)*.

## 2. **User-Based Collaborative Filtering**

- Computes cosine similarity between **users**.
- Suggests movies a user hasn't watched based on ratings from similar users.
- Example: Top 5 recommendations for `User 1`.


Both CF types include prediction functions:

- `predictUserBased(userId, movieId)`
- `predictItemBased(userId, movieId)`

These functions are used to predict unseen ratings for evaluation.

---

## Evaluation

### Metrics:

- **Mean Absolute Error**
- **Root Mean Squared Error**

### Initial Results:

| Method               | MAE     | RMSE    |
|----------------------|---------|---------|
| User-Based CF        | 0.8834  | 41.3271 |
| Item-Based CF        | 0.7087  | 2.3991  |

> **Note**: The extremely high RMSE in User-Based CF was due to unbounded predictions and a lack of filtering.

---

### Improving Performance

- Subset of active users (≥20 ratings) and popular movies (≥50 ratings).
- Limited test set to 10,000 samples for faster evaluation.
- Used **top-K similar users/items (e.g., K=10)**.
- **Filtered out negative similarities**.
- Applied **clipping to predictions (1 to 5)** to reduce extreme prediction errors.
- Handled **cold-start** cases using global mean rating.

### Improved Results (after enhancements):

#### Item-Based Collaborative Filtering:
* The initial model yielded an MAE of 0.7087 and an RMSE of 2.3991.
* The improved model showed significantly better performance with an MAE of 0.6535 and an RMSE of 0.8380.

#### User-Based Collaborative Filtering:
*	The initial model had an MAE of 0.8834 and a high RMSE of 41.3271.
*	The improved model achieved a much lower MAE of 0.6832 and RMSE of 0.8749.

---

## 3. **Matrix Factorization (TruncatedSVD)**

For a latent feature–based approach, I implemented **TruncatedSVD Matrix Factorization**. The rating matrix was decomposed into lower-dimensional latent factors for users and items.

**Method**:

* Built a user-item interaction matrix.
* Applied `TruncatedSVD` to reduce dimensionality and reconstruct the matrix.
* Used reconstructed ratings for predictions.

**Results**:
*	This method resulted in an MAE of 2.4385 and an RMSE of 2.7063.

> This approach performed significantly worse than others. High MAE/RMSE suggests it struggled with data sparsity and lacked normalization.

---

## 4. **Neural Network (Autoencoder)**

I also experimented with a **deep learning approach** using a **simple Autoencoder** architecture to predict missing ratings based on compressed user rating profiles.

**Method**:

* Normalized rating matrix.
* Used a symmetrical autoencoder: compression → bottleneck → reconstruction.
* Trained with MSE loss and dropout for regularization.

**Results**:
*	This approach achieved an MAE of 0.6959 and an RMSE of 0.8856.

> This method performed **better than collaborative filtering and matrix factorization**, capturing complex nonlinear relationships effectively.

---

**Conclusion**: Subsetting data, using top-K filtering, and clipping predictions significantly improve model performance.
