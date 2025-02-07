# Load required library for Shiny
library(shiny)

# Define UI for the application
ui <- fluidPage(
  titlePanel("Bayes Classifier"),
  
  sidebarLayout(
    sidebarPanel(
      fileInput("file1", "Upload Dataset for Class 1"),
      fileInput("file2", "Upload Dataset for Class 2"),
      fileInput("file3", "Upload Dataset for Class 3"),
      radioButtons("covariance", "Covariance Type:",
                   choices = list("Same covariance matrix (σ^2 I)" = "sigma2_I",
                                  "Full covariance matrix (Σ)" = "full_same",
                                  "Diagonal covariance matrix (different for each class)" = "diag_diff",
                                  "Full covariance matrix (different for each class)" = "full_diff")),
      actionButton("run", "Run Classifier")
    ),
    
    mainPanel(
      h3("Confusion Matrix"),
      verbatimTextOutput("confMatrix"),
      
      h3("Evaluation Metrics"),
      verbatimTextOutput("metrics")
    )
  )
)

# Define server logic for the application
server <- function(input, output) {
  
  # Function to compute covariance matrix
  compute_covariance <- function(data) {
    cov_matrix <- cov(data)
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
  
  # React to the run button
  observeEvent(input$run, {
    req(input$file1, input$file2, input$file3)  # Ensure files are uploaded
    
    # Load datasets
    data1 <- read.table(input$file1$datapath, header = FALSE)
    data2 <- read.table(input$file2$datapath, header = FALSE)
    data3 <- read.table(input$file3$datapath, header = FALSE)
    
    # Convert data to numeric and check for NA values
    data1 <- as.data.frame(lapply(data1, function(col) as.numeric(col)))
    data2 <- as.data.frame(lapply(data2, function(col) as.numeric(col)))
    data3 <- as.data.frame(lapply(data3, function(col) as.numeric(col)))
    
    if (any(is.na(data1)) | any(is.na(data2)) | any(is.na(data3))) {
      stop("Data contains NA values. Please check and clean the datasets.")
    }
    
    # Combine data
    data1 <- data.frame(X1 = data1$V1, X2 = data1$V2, Class = 1)
    data2 <- data.frame(X1 = data2$V1, X2 = data2$V2, Class = 2)
    data3 <- data.frame(X1 = data3$V1, X2 = data3$V2, Class = 3)
    all_data <- rbind(data1, data2, data3)
    
    # Split into training and test sets
    set.seed(123)
    n1 <- nrow(data1)
    n2 <- nrow(data2)
    n3 <- nrow(data3)
    train_idx1 <- sample(1:n1, size = 0.7 * n1)
    train_idx2 <- sample(1:n2, size = 0.7 * n2)
    train_idx3 <- sample(1:n3, size = 0.7 * n3)
    
    train_data <- rbind(data1[train_idx1, ], data2[train_idx2, ], data3[train_idx3, ])
    test_data <- rbind(data1[-train_idx1, ], data2[-train_idx2, ], data3[-train_idx3, ])
    
    # Compute means and covariance matrices
    means <- list()
    cov_matrices <- list()
    
    for (i in 1:3) {
      class_data <- train_data[train_data$Class == i, 1:2]
      means[[i]] <- colMeans(class_data)
      cov_matrices[[i]] <- compute_covariance(class_data)
    }
    
    # Covariance matrix selection based on input
    if (input$covariance == "sigma2_I") {
      avg_cov <- (cov_matrices[[1]] + cov_matrices[[2]] + cov_matrices[[3]]) / 3
      cov_matrix <- rep(list(diag(mean(diag(avg_cov)), 2, 2)), 3)
    } else if (input$covariance == "full_same") {
      avg_cov <- (cov_matrices[[1]] + cov_matrices[[2]] + cov_matrices[[3]]) / 3
      cov_matrix <- rep(list(avg_cov), 3)
    } else if (input$covariance == "diag_diff") {
      cov_matrix <- lapply(cov_matrices, function(cov_matrix) diag(diag(cov_matrix)))
    } else if (input$covariance == "full_diff") {
      cov_matrix <- cov_matrices
    }
    
    # Evaluate the model
    results <- evaluate_model(test_data, means, cov_matrix)
    
    # Output confusion matrix and metrics
    output$confMatrix <- renderPrint({ results$conf_matrix })
    output$metrics <- renderPrint({
      cat("Accuracy:", results$accuracy, "\n")
      cat("Precision:", results$precision, "\n")
      cat("Recall:", results$recall, "\n")
      cat("F1 Score:", results$f1, "\n")
    })
  })
}

# Run the application 
shinyApp(ui = ui, server = server)
