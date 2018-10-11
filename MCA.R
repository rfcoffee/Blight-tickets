# load packages
require(FactoMineR)
require(openxlsx)
library(openxlsx)
library(caTools)
require(RcppCNPy)
library(reticulate)
np <-import("numpy")

# define command-line arguments
args = commandArgs(trailingOnly=TRUE)
if (length(args)==0)
{
  args[1] = "2"
} # else if (length(args)==1)
#{
#  temp = paste("train_", args[1], sep="")
#  args[2] = paste(temp, ".xlsx", sep="")
#}
# number of dimensions to keep
ncol_mca = strtoi(args[1])
train_cat = read.xlsx("train_cat.xlsx")
cv_cat = read.xlsx("cv_cat.xlsx")

# convert each columns to type: factor
for (col in colnames(train_cat))
{
  train_cat[,col] <- as.factor(train_cat[,col])
  cv_cat[,col] <- as.factor(cv_cat[,col])
}

# do MCA
mca1 <- MCA(train_cat, ncp=ncol_mca, graph = FALSE)

# project test data
train_mca <- predict(mca1, train_cat)$coord
cv_mca <- predict(mca1, cv_cat)$coord

np$save(paste(paste("train_mca_", args[1], sep=""), ".npy", sep=""), data.matrix(train_mca))
np$save(paste(paste("cv_mca_", args[1], sep=""), ".npy", sep=""), data.matrix(cv_mca))