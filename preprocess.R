# load packages
require(FactoMineR)
require(openxlsx)
library(openxlsx)
library(caTools)
require(RcppCNPy)
library(reticulate)
np <-import("numpy")

# features to include
cat_feats <- c('admin_fee', 'disposition')
num_feats <- c('fine_amount', 'late_fee', 'discount_amount', 'judgment_amount')
target <- c('compliance')
include_feats <- c(cat_feats, num_feats)
include_cols <- c(include_feats, target)

# read files from python, and take 
data0 <- read.csv(file="train.csv", header=TRUE)
data1 <- data0[,include_cols]
data2 <- data1[complete.cases(data1), ]

# split data into train/test
set.seed(123)
data2$spl = sample.split(data2[,1],SplitRatio=2/3)
train <- subset(data2, data2$spl==TRUE)[,include_cols]
cv <- subset(data2, data2$spl==FALSE)[,include_cols]

# save numerical features and target
np$save("train_num.npy", data.matrix(train[,num_feats]))
np$save("train_y.npy", as.vector(train[,target]))
np$save("cv_num.npy", data.matrix(cv[,num_feats]))
np$save("cv_y.npy", as.vector(cv[,target]))

write.xlsx(train[,cat_feats], "train_cat.xlsx")
write.xlsx(cv[,cat_feats], "cv_cat.xlsx")
