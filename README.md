# H_S_comp
### Goal
# predict the sales price for each house.
# RMSE is used to evaluate performance (Kaggle leaderboard eval uses log of 
# SalePrice to treat high and low price houses equally)

library(GGally)
library(MASS)
library(gbm)
library(glmnet)
library(randomForest)
library(xgboost)
library(vtreat)
library(SuperLearner)
library(Hmisc)
library(Metrics)
library(usethis)
library(ranger)
library(caret)
library(Hmisc)
library(corrplot)
library(mice)
library(aCRM)
library(OpenMPController)
library(e1071) 
library(dplyr)
library(tidyverse)
options(scipen=999)
set.seed(57)
omp_set_num_threads(4) # caret parallel processing threads

### could also use eval functions from Metrics packages
### build calc function for RMSE
rmse_fun <- function(actual, predicted) {
            sqrt(mean((predicted - actual)^2))
}

################################################################################
### read in the data
################################################################################
houses_train_df_raw <- read_csv("train.csv")
houses_train_df_raw <- houses_train_df_raw %>% mutate(label = "train")

houses_test_df <- read_csv("test.csv")
houses_test_df <- houses_test_df %>% mutate(label = "test")

### removing some of the outliers as linear regression models in the ensemble will be impacted
### include training examples which are less than 6 st devs from the mean
houses_train_df <- houses_train_df %>%
            mutate(total_building_SF =  dplyr::select(., `1stFlrSF`, `2ndFlrSF`, TotalBsmtSF, GarageArea) %>% rowSums(na.rm = TRUE))

### filter here removes outliers in the training data used for models
houses_train_df <- houses_train_df %>%
            mutate(total_building_SF_zscore = (total_building_SF - mean(houses_train_df$total_building_SF))/
                    sd(houses_train_df$total_building_SF)) %>%
            filter(total_building_SF_zscore <= 6) %>%
            dplyr::select(-total_building_SF_zscore, -total_building_SF)

### join train and test data together and do feature cleaning / feature engineering
complete_data <- rbind(dplyr::select(houses_train_df,-SalePrice),
                       houses_test_df)

### adjust feature names which start with numerics
complete_data <- complete_data %>% rename(first_FlrSF = `1stFlrSF`, 
                        second_FlrSF = `2ndFlrSF`, 
                        three_SsnPorch = `3SsnPorch`)

### 4 outliers removed from train data
paste0("Number of outliers removed in the training data: ", nrow(houses_train_df_raw) - nrow(houses_train_df))

################################################################################
### Basic eda
################################################################################
### bunch of NAs in the dataset which we'll need to deal
### with in order to keep training data volume
glimpse(houses_train_df)

summary(houses_train_df)

### distribution of training data sale prices
### some outliers in the upper tail
histogram(houses_train_df$SalePrice/100000)

### some neighborhoods have higher avg priced houses as one would expect
houses_train_df %>% group_by(Neighborhood) %>%
            summarise(avg_Sales_price = mean(SalePrice)) %>%
            ggplot(aes(x=reorder(Neighborhood,avg_Sales_price),
                       y=avg_Sales_price/100000)) +
            geom_col() +
            coord_flip() +
            labs(y="Average Sale Price (hundred thousand dollars",
                 x="Neighborhood")

### Overall Qual looks to correlate more with SalePrice vs Overall Cond
houses_train_df %>%
            ggplot(aes(x=OverallQual, y=OverallCond, color=cut2(SalePrice,g=6))) +
            geom_jitter(alpha=0.4) +
            facet_grid(. ~ cut2(SalePrice,g=6)) +
            theme(legend.position = "none")

### corrplot on all the numeric features
### need to explore further after cleaning the NAs
numerics_houses_train_df <- houses_train_df %>% select_if(is.numeric)
corrplot(cor(numerics_houses_train_df,use="pairwise"),number.cex= 5/ncol(numerics_houses_train_df))

### correlation matric which outputs numerics
rcorr(as.matrix(numerics_houses_train_df))

################################################################################
### handle missing data
################################################################################

### missing data function to see the features that have NAs
missing_data_fun <- function(df) {
            df_result <- as.data.frame(sapply(df, function(x) sum(is.na(x))))
            df_result <- rownames_to_column(df_result, var = "feature")
            colnames(df_result)[2] <- c("NA_Count")
            df_result %>% filter(NA_Count>0) %>% arrange(desc(NA_Count))
}
missing_data_fun(complete_data)

### When Garage Type is valid, fill in garage year built year as remodel year when missing
complete_data <- complete_data %>%
            mutate(GarageYrBlt = ifelse(is.na(GarageYrBlt) & 
                                                    !is.na(GarageType) & 
                                                    GarageCars>0, 
                                        YearRemodAdd, GarageYrBlt))

### Handle mismatch between Garage Type being non null but no garage cars value set to NA
complete_data <- complete_data %>%
            mutate(GarageType = ifelse(!is.na(GarageType) & is.na(GarageCars),NA,GarageType))

# show mode imputation for categorial data
imputeMissings(complete_data %>%  dplyr::select(Id, contains("Gar"), Id)) %>% filter(Id==2127)

### then hand code missing values for problem house which has partial garage info
### Future todo: implement more elegant way to do this
complete_data[which(complete_data$Id == 2127), 
               which(colnames(complete_data)=="GarageFinish")] <- "Unf"

complete_data[which(complete_data$Id == 2127), 
               which(colnames(complete_data)=="GarageQual")] <- "TA"

complete_data[which(complete_data$Id == 2127), 
               which(colnames(complete_data)=="GarageCond")] <- "TA"

### NAs are coded in the data description but
### represent that the house feature is not present which is useful for prediction
features_to_replace_NA <- c("PoolQC", "MiscFeature", "Alley", "Fence", "FireplaceQu", 
                            "GarageType", "GarageFinish", "GarageQual","GarageCond",
                            "BsmtExposure", "BsmtFinType2", "BsmtCond",
                            "BsmtFinType1", "BsmtQual", "MasVnrType")

### for the above features which contain NAs set NAs 
### to None to signal the house attribute not being present
complete_data <- complete_data %>% mutate_at(features_to_replace_NA, ~replace_na(., "None"))

### Replace NA MasVnrArea with 0
complete_data <- complete_data %>% 
            mutate(MasVnrArea = replace_na(MasVnrArea,0))

### Replace NA GarageYrBlt with 0 after creating Garage dummy var
complete_data <- complete_data %>% 
            mutate(Garage_dummy_flag = ifelse(GarageType=="None", 0, 1),
                   GarageYrBlt = ifelse(is.na(GarageYrBlt), 0, GarageYrBlt))

### Replace NA LotFrontage with neighborhood median
### Future todo: implement more elegant way to do this 
### i.e. use prediction to predict the missing value
median_LotFrontage_df <- complete_data %>% 
            group_by(Neighborhood) %>% 
            summarise(median_LotFrontage = median(LotFrontage, na.rm = TRUE))

complete_data <- left_join(complete_data, median_LotFrontage_df, by=c("Neighborhood")) %>%
            mutate(LotFrontage = ifelse(is.na(LotFrontage),median_LotFrontage,LotFrontage)) %>%
             dplyr::select(-median_LotFrontage)

### Update house 949 with most common neighborhood BsmtExposure
CollgCr_BsmtExposure_most_common <- complete_data %>% 
            filter(BsmtExposure!="None") %>%
            filter(Neighborhood=="CollgCr") %>%
            group_by(Neighborhood, BsmtExposure) %>% 
            count() %>%
            ungroup() %>%
            arrange(desc(n)) %>%
             dplyr::select(BsmtExposure) %>%
            head(1)
CollgCr_BsmtExposure_most_common[[1]]
            
complete_data[which(complete_data$Id == 949), 
                which( colnames(complete_data)=="BsmtExposure")] <- CollgCr_BsmtExposure_most_common[[1]]

### House Id 1380 is showing up as not having Electrical but has heating and cooling
### recode as having Electral 
Electral_most_common_2006_houses <- complete_data %>% 
            filter(YearBuilt==2006) %>%
            group_by(Electrical) %>%
            count() %>% 
            arrange(desc(n)) %>% 
            head(1)

complete_data[which(complete_data$Id == 1380), 
                which(colnames(complete_data)=="Electrical")] <- Electral_most_common_2006_houses[[1]]

### 4 obs with MSZoning missing
### find most common MSZoning by neighborhood
most_common_MSZone <- complete_data %>%
            group_by(Neighborhood, MSZoning) %>%
            count() %>%
            arrange(desc(n)) %>%
            ungroup() %>%
            group_by(Neighborhood) %>%
            mutate(rank = row_number()) %>%
            filter(rank==1) %>%
            ungroup() %>%
             dplyr::select(-n, -rank) %>%
            rename(agg_MSZoning = MSZoning)

complete_data <- left_join(complete_data, most_common_MSZone, by=c("Neighborhood")) %>%
            mutate(MSZoning = ifelse(is.na(MSZoning),agg_MSZoning,MSZoning)) %>%
             dplyr::select(-agg_MSZoning)

### Final hand cleaning of features
### Future todo: use more dynamic approach which could better handle input data changing
complete_data <- complete_data %>% 
            mutate(Utilities = ifelse(is.na(Utilities),"AllPub", Utilities),
                   BsmtFullBath = ifelse(is.na(BsmtFullBath),0, BsmtFullBath),
                   BsmtHalfBath = ifelse(is.na(BsmtHalfBath),0, BsmtHalfBath),
                   Functional = ifelse(is.na(Functional),'Typ', Functional), 
                   Exterior1st = ifelse(is.na(Exterior1st),'MetalSd', Exterior1st), 
                   Exterior2nd = ifelse(is.na(Exterior2nd),Exterior1st, Exterior2nd),
                   BsmtFinSF1 = ifelse(BsmtCond=="None",0, BsmtFinSF1), 
                   BsmtFinSF2 = ifelse(BsmtCond=="None",0, BsmtFinSF2), 
                   BsmtUnfSF = ifelse(BsmtCond=="None",0, BsmtUnfSF),
                   TotalBsmtSF = ifelse(BsmtCond=="None",0, TotalBsmtSF), 
                   KitchenQual = ifelse(is.na(KitchenQual),"TA", KitchenQual),
                   GarageCars = ifelse(GarageType=="None",0, GarageCars),
                   GarageArea = ifelse(GarageType=="None",0, GarageArea),
                   SaleType = ifelse(is.na(SaleType),"WD",SaleType))

### NAs now cleaned from complete data set
missing_data_fun(complete_data)

################################################################################
### Feature engineering/scaling/normalizing
################################################################################

### create derived features
complete_data <- complete_data %>% 
            mutate(total_building_SF_non_porch = first_FlrSF + second_FlrSF + TotalBsmtSF + GarageArea,
                   total_porch_and_deck_SF = WoodDeckSF + OpenPorchSF + EnclosedPorch + three_SsnPorch + ScreenPorch,
                   OverallQual_plus_Cond = OverallQual + OverallCond,
                   total_bathrooms = BsmtFullBath + (BsmtHalfBath*0.5) + FullBath + (HalfBath*0.5),
                   years_between_build_and_remodel = YearRemodAdd - YearBuilt,
                   number_of_non_bedrooms = TotRmsAbvGrd - BedroomAbvGr)

### one hot encode using vtreat
complete_data <- complete_data %>% mutate_if(is.character,as.factor)
varlist <- colnames(complete_data %>%  dplyr::select(-label))
treatplan <- designTreatmentsZ(complete_data, varlist)
scoreFrame <- treatplan %>% 
            magrittr::use_series(scoreFrame) %>% 
            dplyr::select(varName, origName, code)

### this drops some features types not in code list
newvars <- scoreFrame %>%
            filter(code %in% c("clean", "lev", "isBAD")) %>%
            use_series(varName)

data.treat <- prepare(treatplan, complete_data, varRestriction = newvars)

complete_data_labels <- complete_data %>%  dplyr::select(Id, label)

### join train and test labels back to complete dataset which is now one hot encoded
complete_data <- left_join(data.treat, complete_data_labels, by=c("Id"))

### take the log of features that have high skew
### find numeric columns to calculate skew for
numeric_columns <- select_if(complete_data, is.numeric) %>%
            dplyr::select(-Id) %>%
            colnames()

skew_df <- as.data.frame(
            sapply(complete_data %>% select_(.dots = numeric_columns), 
                   function(x) skewness(x)
      )
)

skew_df <- rownames_to_column(skew_df, var = "feature") %>% rename(skew = 2)

features_to_log_transform <- filter(skew_df, skew > 2.5 | skew < -2.5) %>% 
            dplyr::select(feature)
features_to_log_transform <- c(features_to_log_transform$feature)

### log transform features which have high skew
complete_data <- complete_data %>% 
            mutate_at(vars(features_to_log_transform), 
                                 function(x) log1p(x))

################################################################################
### Model Building
################################################################################

### Join in target variable back to the training data
final_houses_train_df <- complete_data %>% filter(label=="train")
final_houses_train_df <- final_houses_train_df %>% 
            left_join(dplyr::select(houses_train_df, Id, SalePrice), by=c("Id"))
final_houses_train_df <- final_houses_train_df %>% 
            ### predict the log of the SalePrice which meetings normal distribution assumptions
            mutate(SalePrice_log = log1p(SalePrice)) %>%
            dplyr::select(-Id,-SalePrice,-label)

### Use ranger RF model to filter down the feature space
ranger_model_for_feature_selection <- ranger(formula = SalePrice_log ~ .,
                       importance = 'impurity',
                       data    = final_houses_train_df)

rmse_fun(final_houses_train_df$SalePrice_log,
         predict(ranger_model_for_feature_selection, data=final_houses_train_df)$predictions)

feature_importance <- data.frame(importance_score = ranger_model_for_feature_selection$variable.importance)
feature_importance <- rownames_to_column(feature_importance, var = "features")
feature_importance <- feature_importance %>%
            arrange(desc(importance_score)) %>%
            mutate(feature_rank = row_number()) %>%
            filter(feature_rank<=200)

### selects the features that meet name criteria in the append set
final_houses_train_df <- final_houses_train_df %>%
            select_(.dots = append(feature_importance$features,"SalePrice_log"))

### Out of the box ranger model on the paired down feature set
ranger_model <- ranger(formula = SalePrice_log ~ .,
                         importance = 'impurity',
                         data = final_houses_train_df)
### ranger rf rmse
rmse_fun(final_houses_train_df$SalePrice_log, 
         predict(ranger_model, data=final_houses_train_df)$predictions)

### XGB model using caret
xgb_grid <- expand.grid(
            nrounds = 500,
            eta = 0.3,
            max_depth = 2, 
            gamma=0,
            colsample_bytree = 1,
            min_child_weight = 1,
            subsample = 1)

train_control <- caret::trainControl(
            method = "none",
            verboseIter = FALSE,
            allowParallel = TRUE)

input_x <- as.matrix(final_houses_train_df %>%  dplyr::select(-SalePrice_log))
input_y <- final_houses_train_df$SalePrice_log

xgb_model <- caret::train(
            x = input_x,
            y = input_y,
            trControl = train_control,
            tuneGrid = xgb_grid,
            #early_stopping_rounds = 250,
            method = "xgbTree",
            verbose = FALSE)

### View most important XGB features
head(varImp(xgb_model)$importance, 30) %>%
            rownames_to_column(var = "feature") %>%
            ggplot(aes(x=reorder(feature,Overall), y=Overall, fill=Overall)) +
            geom_col() +
            coord_flip() +
            theme(legend.position = "none") +
            labs(subtitle = "Top 30 Most Important XGB Features")

### XGB rmse
rmse_fun(final_houses_train_df$SalePrice_log, predict(xgb_model))

### lambdas for ridge and lasso
lambdas_grid <- 10^seq(10, -2, length = 100)

### Ridge regression
ridge_regression <- glmnet(input_x, input_y, alpha = 0, lambda = lambdas_grid)
cv_ridge_regression <- cv.glmnet(input_x, input_y, alpha = 0, lambda = lambdas_grid)
best_lambda_ridge <- cv_ridge_regression$lambda.min

### Ridge rmse
rmse_fun(final_houses_train_df$SalePrice_log, 
         predict(ridge_regression, s = best_lambda_ridge, newx = input_x))

### Lasso regression
lasso_regression <- glmnet(input_x, input_y, alpha = 1, lambda = lambdas_grid)
cv_lasso_regression <- cv.glmnet(input_x, input_y,  alpha = 1, lambda = lambdas_grid)
best_lambda_lasso <- cv_lasso_regression$lambda.min

### Lasso rmse
rmse_fun(final_houses_train_df$SalePrice_log, 
         predict(lasso_regression, s = best_lambda_lasso, newx = input_x))

### investigate where there is highest disagreement between the models
compare_models_predictions <- tibble(
            ranger = predict(ranger_model, data=final_houses_train_df)$predictions,
            xgb = predict(xgb_model),
            ridge = predict(ridge_regression, s = best_lambda_ridge, newx = input_x)[,1],
            lasso = predict(lasso_regression, s = best_lambda_lasso, newx = input_x)[,1]
)

head(predict(lasso_regression, s = best_lambda_lasso, newx = input_x)[,1])

### now run apply functions on df with the preds only and add to compare models df
compare_models <- compare_models_predictions
compare_models$avg_pred <- apply(compare_models_predictions,1, mean, na.rm = TRUE)
compare_models$median_pred <- apply(compare_models_predictions,1, median, na.rm = TRUE)
compare_models$pred_standard_dev <- apply(compare_models_predictions,1, sd, na.rm = TRUE)
compare_models$SalePrice_log <- final_houses_train_df$SalePrice_log
compare_models$SalePrice_log_scaled <- scale(compare_models$SalePrice_log)
compare_models$residuals_avg_pred <- compare_models$SalePrice_log - compare_models$avg_pred
compare_models$residuals_median_pred <- compare_models$SalePrice_log - compare_models$median_pred
compare_models$lasso_resid <- compare_models$SalePrice_log - compare_models_predictions$lasso
compare_models$ridge_resid <- compare_models$SalePrice_log - compare_models_predictions$ridge
compare_models$ranger_resid <- compare_models$SalePrice_log - compare_models_predictions$ranger
compare_models$xgb_resid <- compare_models$SalePrice_log - compare_models_predictions$xgb

### visually expect disagreement and can look up in train data to spot potential improvements
# compare_models %>% View()

### plot showing ensemble residuals on training data
plot(exp(compare_models$SalePrice_log), compare_models$residuals_avg_pred, 
       ylab="Residuals", xlab="Sale Price Log", 
       main="Resid Plot Sales Price Log Predictions") 
abline(0, 0)                

### 4 plot grid showing Residuals by Model Type on Training Data
compare_models %>% select(SalePrice_log, lasso_resid, 
                          ridge_resid, xgb_resid, ranger_resid) %>%
            gather(key = model, value = resid, lasso_resid, 
                   ridge_resid, xgb_resid, ranger_resid) %>%
            ggplot(aes(x= SalePrice_log, y=resid, color=model)) +
            geom_point(alpha=0.4) +
            geom_hline(yintercept = 0, linetype="dashed", alpha=0.8, color="grey40") +
            facet_wrap(. ~ model, ncol=2) +
            labs(title = "Residuals by Model Type on Training Data")

################################################################################
### Predict on test data
################################################################################
test_data <- complete_data %>% filter(label=="test") %>%
             dplyr::select(-label, -Id) %>%
             select_(.dots = feature_importance$features)

test_preds_ranger <- exp(predict(ranger_model, data=test_data)$predictions)
test_preds_xgboost <- exp(predict(xgb_model, newdata =as.matrix(test_data)))
test_preds_ridge <- exp(predict(ridge_regression, s=best_lambda_ridge, newx=as.matrix(test_data))[,1])
test_preds_lasso <- exp(predict(lasso_regression, s=best_lambda_lasso, newx=as.matrix(test_data))[,1])

results_df <- tibble(
            ranger = test_preds_ranger,
            xgb = test_preds_xgboost,
            ridge = test_preds_ridge,
            lasso = test_preds_lasso
)

### take the mean of the predictions from each model as the final test prediction
test_preds <- apply(results_df, 1, function(x) mean(x))

### build test sub df
test_sub <- tibble(Id=houses_test_df$Id, SalePrice=test_preds)

### output test sub in Kaggle requested format
today <- Sys.time()
time_string <- format(today, format="%B_%d_%Y_%H_%M")
write_csv(test_sub, paste0("submission_",time_string,".csv"))
print(paste0("submission_",time_string,".csv"))
read_csv(paste0("submission_",time_string,".csv"))
