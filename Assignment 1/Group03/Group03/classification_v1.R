# Load the datasets from txt files
class1 <- read.table("LS_Group03/Class1.txt")
class2 <- read.table("LS_Group03/Class2.txt")
class3 <- read.table("LS_Group03/Class3.txt")

# Since data has 3 classes, 
# Data is 2D, with the third column indicating the class label
data1 <- data.frame(X1 = class1$V1, X2 = class1$V2, Class = 1)
data2 <- data.frame(X1 = class2$V1, X2 = class2$V2, Class = 2)
data3 <- data.frame(X1 = class3$V1, X2 = class3$V2, Class = 3)

# Combine all datasets into one
all_data <- rbind(data1, data2, data3)

# Function to split data into training and test sets (70/30 split)
split_data <- function(data) {
  set.seed(123)  # For reproducibility
  n <- nrow(data)
  train_idx <- sample(1:n, size = 0.7 * n)
  train_data <- data[train_idx, ]
  test_data <- data[-train_idx, ]
  return(list(train = train_data, test = test_data))
}

# Splitting data by class
train_data1 <- split_data(data1)$train
test_data1 <- split_data(data1)$test

train_data2 <- split_data(data2)$train
test_data2 <- split_data(data2)$test

train_data3 <- split_data(data3)$train
test_data3 <- split_data(data3)$test

# Combine training and test data
train_data <- rbind(train_data1, train_data2, train_data3)
test_data <- rbind(test_data1, test_data2, test_data3)

# Function to compute covariance matrix
compute_covariance <- function(data) {
  n <- nrow(data)
  mu <- colMeans(data)
  cov_matrix <- (t(data) %*% data) / n - mu %*% t(mu)
  return(cov_matrix)
}

# Function to compute Gaussian likelihood
gaussian_likelihood <- function(x, mean, cov_matrix) {
  n <- length(x)
  det_cov <- det(cov_matrix)
  inv_cov <- solve(cov_matrix)
  exp_term <- exp(-0.5 * t(x - mean) %*% inv_cov %*% (x - mean))
  likelihood <- (1 / sqrt((2 * pi)^n * det_cov)) * exp_term
  return(likelihood)
}

# Compute means and covariances
means <- list()
cov_matrices <- list()

for (i in 1:3) {
  class_data <- train_data[train_data$Class == i, 1:2]
  means[[i]] <- colMeans(class_data)
  cov_matrices[[i]] <- compute_covariance(class_data)
}

# 1. Covariance matrix same for all classes (σ^2 I)
avg_cov <- (cov_matrices[[1]] + cov_matrices[[2]] + cov_matrices[[3]]) / 3
cov_sigma2_I <- diag(mean(diag(avg_cov)), 2, 2)

# 2. Full covariance matrix same for all classes (Σ)
cov_full_same <- avg_cov

# 3. Diagonal covariance matrix different for each class
cov_diag <- lapply(cov_matrices, function(cov_matrix) diag(diag(cov_matrix)))

# 4. Full covariance matrix different for each class
cov_full_diff <- cov_matrices

# Function to predict class
predict_class <- function(x, means, cov_matrix) {
  likelihoods <- numeric(length(means))
  for (i in 1:length(means)) {
    likelihoods[i] <- gaussian_likelihood(x, means[[i]], cov_matrix[[i]])
  }
  return(which.max(likelihoods))
}

# Function to evaluate model
evaluate_model <- function(test_data, means, cov_matrix) {
  predictions <- numeric(nrow(test_data))
  for (i in 1:nrow(test_data)) {
    predictions[i] <- predict_class(as.numeric(test_data[i, 1:2]), means, cov_matrix)
  }
  
  true_labels <- test_data$Class
  conf_matrix <- table(Predicted = predictions, Actual = true_labels)
  
  accuracy <- sum(diag(conf_matrix)) / sum(conf_matrix)
  precision <- diag(conf_matrix) / rowSums(conf_matrix)
  recall <- diag(conf_matrix) / colSums(conf_matrix)
  f1 <- 2 * precision * recall / (precision + recall)
  
  return(list(conf_matrix = conf_matrix, accuracy = accuracy, 
              precision = precision, recall = recall, f1 = f1))
}

# Evaluate each classifier
results_sigma2_I <- evaluate_model(test_data, means, rep(list(cov_sigma2_I), 3))
results_full_same <- evaluate_model(test_data, means, rep(list(cov_full_same), 3))
results_diag <- evaluate_model(test_data, means, cov_diag)
results_full_diff <- evaluate_model(test_data, means, cov_full_diff)

# Display results
print("Results for Covariance matrix same for all classes (σ^2 I):")
print(results_sigma2_I)

print("Results for Full covariance matrix same for all classes (Σ):")
print(results_full_same)

print("Results for Diagonal covariance matrix different for each class:")
print(results_diag)

print("Results for Full covariance matrix different for each class:")
print(results_full_diff)
