#
# R/calibration.R
# Calibration helpers for binary classifiers (R).
# Implements Platt scaling: a logistic regression that calibrates raw
# predicted probabilities to better match observed frequencies.
# Always returns a safe mapping function; falls back to identity when the
# data are not suitable (single-class labels) or the fit fails.
#

# Clip probabilities away from exact 0/1 to avoid numerical issues
clip01 <- function(x) pmin(pmax(x, 1e-6), 1 - 1e-6)

# Train a Platt scaler on raw probabilities `p` and binary labels `y`
platt_train <- function(p, y){
  p <- clip01(p)
  y <- as.integer(y)
  if (length(unique(y)) < 2L) return(list(map = function(z) z, method="identity"))
  
  # Fit logistic regression: y ~ p
  fit <- tryCatch(glm(y ~ p, family = binomial()), error=function(e) NULL)
  if (is.null(fit)) return(list(map = function(z) z, method="identity"))
  
  # Return a mapping function that applies the learned calibration
  list(
    map = function(z){
      z <- clip01(z)
      stats::predict(fit, newdata = data.frame(p = z), type = "response")
    },
    method = "platt", model = fit
  )
}
