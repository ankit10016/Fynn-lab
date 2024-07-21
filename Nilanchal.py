import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.linear_model import LinearRegression
from sklearn.mixture import GaussianMixture

# Load the dataset
data = pd.read_csv(r"C:\Users\Nilanchal Mohanty\Downloads\mcdonalds.csv")

# Display initial data information
print(data.columns)
print(data.shape)
print(data.head(5))
print(data.isnull().sum())
print(data.info())
print(data['Like'].unique())

# Mapping 'Like' column to integers
mapping = {
    "I hate it!-5": -5,
    "-4": -4,
    "-3": -3,
    "-2": -2,
    "-1": -1,
    "0": 0,
    "+1": 1,
    "+2": 2,
    "+3": 3,
    "+4": 4,
    "I love it!+5": 5
}
data['Like'] = data['Like'].replace(mapping)
print(data['Like'].unique())

# Select the relevant columns (first 11 columns)
MD_x = data.iloc[:, 0:11]

# Convert "Yes" to 1 and "No" to 0
MD_x = MD_x.applymap(lambda x: 1 if x == "Yes" else 0)

# Display column means
col_means = MD_x.mean().round(2)
print(col_means)

# Standardize the data
scaler = StandardScaler()
MD_x_scaled = scaler.fit_transform(MD_x)

# Perform PCA
pca = PCA(n_components=11)
MD_x_pca = pca.fit_transform(MD_x_scaled)

# Extract the PCA rotation matrix (loadings)
loadings = pca.components_.T

# Create a DataFrame to display the rotation matrix
rotation_df = pd.DataFrame(loadings, index=MD_x.columns, columns=[f'PC{i+1}' for i in range(loadings.shape[1])])
print(rotation_df.round(3))

# Reduce to 2 principal components for visualization
pca = PCA(n_components=2)
MD_x_pca = pca.fit_transform(MD_x_scaled)

# Plot PCA projections
plt.figure(figsize=(10, 7))
plt.scatter(MD_x_pca[:, 0], MD_x_pca[:, 1], color='grey', alpha=0.5)
plt.title('PCA Projections')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')

# Plot PCA axes
for i in range(pca.components_.shape[0]):
    plt.arrow(0, 0, pca.components_[i, 0], pca.components_[i, 1],
              color='red', alpha=0.7, head_width=0.05)
    plt.text(pca.components_[i, 0], pca.components_[i, 1], MD_x.columns[i],
             color='red', fontsize=12, ha='center')

plt.grid(True)
plt.show()

### KMeans Clustering

inertia_values = []

for k in range(2, 9):
    kmeans = KMeans(n_clusters=k, random_state=1234, n_init=10)
    kmeans.fit(MD_x_scaled)
    inertia_values.append(kmeans.inertia_)

# Plot the inertia values against the number of clusters
plt.figure(figsize=(10, 7))
plt.plot(range(2, 9), inertia_values, marker='o')
plt.title('KMeans Clustering: Inertia vs. Number of Clusters')
plt.xlabel('Number of Clusters')
plt.ylabel('Inertia')
plt.xticks(range(2, 9))
plt.grid(True)
plt.show()

### Apply Regression Model

# Create Like.n
data['Like.n'] = 6 - data['Like'].astype(float)

# Define independent variables and dependent variable
X = MD_x[['yummy', 'convenient', 'spicy', 'fattening', 'greasy', 'fast', 'cheap', 'tasty', 'expensive', 'healthy', 'disgusting']]
y = data['Like.n']

# Fit Gaussian Mixture Model
gmm = GaussianMixture(n_components=2, random_state=1234)
gmm.fit(X)
components = gmm.predict(X)

# Assign components to the original data
data['Component'] = components

# Fit Linear Regression Models for Each Component
models = {}

for component in range(2):
    # Filter data for the current component
    X_component = X[data['Component'] == component]
    y_component = y[data['Component'] == component]
    
    # Fit linear regression model
    model = LinearRegression()
    model.fit(X_component, y_component)
    
    # Store model
    models[component] = model

    # Print coefficients
    print(f"Component {component} coefficients:")
    print("Intercept:", model.intercept_)
    print("Coefficients:", model.coef_)

# Visualize regression coefficients for each component
coef_df = pd.DataFrame.from_dict(
    {component: model.coef_ for component, model in models.items()},
    orient='index',
    columns=X.columns
)
coef_df['Intercept'] = [model.intercept_ for model in models.values()]

# Plot coefficients
coef_df.T.plot(kind='bar', figsize=(12, 8))
plt.title('Regression Coefficients by Component')
plt.xlabel('Features')
plt.ylabel('Coefficient Value')
plt.legend(title='Component')
plt.grid(True)
plt.show()
