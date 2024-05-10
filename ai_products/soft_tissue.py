import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Load the dataset
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer-wisconsin/wdbc.data"
column_names = ['ID', 'Diagnosis', 'Mean Radius', 'Mean Texture', 'Mean Perimeter', 'Mean Area', 'Mean Smoothness',
                'Mean Compactness', 'Mean Concavity', 'Mean Concave Points', 'Mean Symmetry', 'Mean Fractal Dimension',
                'SE Radius', 'SE Texture', 'SE Perimeter', 'SE Area', 'SE Smoothness', 'SE Compactness', 'SE Concavity',
                'SE Concave Points', 'SE Symmetry', 'SE Fractal Dimension', 'Worst Radius', 'Worst Texture', 'Worst Perimeter',
                'Worst Area', 'Worst Smoothness', 'Worst Compactness', 'Worst Concavity', 'Worst Concave Points',
                'Worst Symmetry', 'Worst Fractal Dimension']
data = pd.read_csv(url, names=column_names, header=None)

# Preprocess the data
X = data.drop(columns=['ID', 'Diagnosis'])
y = (data['Diagnosis'] == 'M').astype(int)

# Split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define evaluation function
def evaluate_model(individual):
    # Individual is a set of hyperparameters
    n_estimators = individual[0]
    max_depth = individual[1]

    # Train RandomForestClassifier with given hyperparameters
    model = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, random_state=42)
    model.fit(X_train, y_train)

    # Evaluate on test set
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    return accuracy

# Genetic Algorithm
class GeneticAlgorithm:
    def __init__(self, pop_size, n_gen, mutation_rate):
        self.pop_size = pop_size
        self.n_gen = n_gen
        self.mutation_rate = mutation_rate

    def init_population(self):
        return [[np.random.randint(10, 100), np.random.randint(1, 10)] for _ in range(self.pop_size)]

    def crossover(self, parent1, parent2):
        # Single-point crossover
        crossover_point = np.random.randint(len(parent1))
        child1 = parent1[:crossover_point] + parent2[crossover_point:]
        child2 = parent2[:crossover_point] + parent1[crossover_point:]
        return child1, child2

    def mutate(self, individual):
        # Mutate each gene with a certain probability
        for i in range(len(individual)):
            if np.random.rand() < self.mutation_rate:
                individual[i] = np.random.randint(10, 100) if i == 0 else np.random.randint(1, 10)
        return individual

    def evolve(self):
        population = self.init_population()
        for _ in range(self.n_gen):
            # Evaluate fitness of each individual
            fitness_scores = [evaluate_model(individual) for individual in population]
            # Select parents based on fitness scores
            parents = [population[i] for i in np.argsort(fitness_scores)[-2:]]
            # Generate offspring through crossover and mutation
            offspring = [self.mutate(child) for child in self.crossover(*parents)]
            # Replace least fit individuals with offspring
            population = population[:-2] + offspring
        # Return the best individual
        return max(population, key=evaluate_model)

# Hyperparameters
pop_size = 10
n_gen = 20
mutation_rate = 0.1

# Initialize and run Genetic Algorithm
ga = GeneticAlgorithm(pop_size, n_gen, mutation_rate)
best_hyperparams = ga.evolve()

print("Best hyperparameters found by Genetic Algorithm:", best_hyperparams)

# After the best hyperparameters are found
best_n_estimators, best_max_depth = best_hyperparams

# Train RandomForestClassifier with the best hyperparameters
best_model = RandomForestClassifier(n_estimators=best_n_estimators, max_depth=best_max_depth, random_state=42)
best_model.fit(X_train, y_train)

# Predict on test set
y_pred = best_model.predict(X_test)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy on test set:", accuracy)
