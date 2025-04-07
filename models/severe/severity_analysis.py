from pathlib import Path, PurePath

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.metrics import roc_curve
from sklearn import preprocessing
from scipy.stats import norm
import sklearn.metrics as metrics
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import StandardScaler
from deriva_ml import DatasetBag
from eye_ai.eye_ai import EyeAI
from feature_engine.encoding import OneHotEncoder


class Severity(EyeAI):
    """
    """

    def __init__(
        self,
        hostname = 'www.eye-ai.org',
        catalog_id = "eye-ai",
        cache_dir: str = "/data",
        working_dir: str = None,
        method="chart_label"
    ):
        """
        Initializes the EyeAI object.

        Args:
        - hostname (str): The hostname of the server where the catalog is located.
        - catalog_number (str): The catalog number or name.
        """

        super().__init__(
            hostname=hostname,
            catalog_id=catalog_id,
            cache_dir=cache_dir,
            working_dir=working_dir,
        )

    def plot_roc(self, configuration_record, data: pd.DataFrame) -> Path:
        """
        Plot Receiver Operating Characteristic (ROC) curve based on prediction results. Save the plot values into a csv file.

        Parameters:
        - data (pd.DataFrame): DataFrame containing prediction results with columns 'True Condition_Label' and
        'Probability Score'.
        Returns:
            Path: Path to the saved csv file of ROC plot values .

        """
        output_path = configuration_record.execution_assets_path("ROC")
        pred_result = pd.read_csv(data)
        y_true = pred_result["True Label"]
        scores = pred_result["Probability Score"]
        fpr, tpr, thresholds = roc_curve(y_true, scores)
        roc_df = pd.DataFrame({"False Positive Rate": fpr, "True Positive Rate": tpr})
        roc_csv_path = output_path / Path("roc_plot.csv")
        roc_df.to_csv(roc_csv_path, index=False)
        # show plot in notebook
        plt.plot(fpr, tpr)
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title("ROC Curve")
        plt.show()

        return roc_csv_path

    def severity_analysis(self, ds_bag: DatasetBag):
        wide = self.multimodal_wide(ds_bag)

        def compare_sides_severity(
            group, value_col, new_col, smaller=True
        ):  # helper method for severity_analysis
            group[new_col] = group[new_col].astype(str)

            if len(group) == 2:  # Ensure there are both left and right sides
                left = group[group["Image_Side"] == "Left"]
                right = group[group["Image_Side"] == "Right"]
                if not left.empty and not right.empty:
                    left_value = left[value_col].values[0]
                    right_value = right[value_col].values[0]
                    if smaller:
                        if left_value < right_value:
                            group.loc[group["Image_Side"] == "Left", new_col] = "Left"
                            group.loc[group["Image_Side"] == "Right", new_col] = "Left"
                        elif left_value == right_value:
                            group.loc[group["Image_Side"] == "Left", new_col] = (
                                "Left/Right"
                            )
                            group.loc[group["Image_Side"] == "Right", new_col] = (
                                "Left/Right"
                            )
                        else:
                            group.loc[group["Image_Side"] == "Left", new_col] = "Right"
                            group.loc[group["Image_Side"] == "Right", new_col] = "Right"
                    else:
                        # Larger value means more severe
                        if left_value > right_value:
                            group.loc[group["Image_Side"] == "Left", new_col] = "Left"
                            group.loc[group["Image_Side"] == "Right", new_col] = "Left"
                        elif left_value == right_value:
                            group.loc[group["Image_Side"] == "Left", new_col] = (
                                "Left/Right"
                            )
                            group.loc[group["Image_Side"] == "Right", new_col] = (
                                "Left/Right"
                            )
                        else:
                            group.loc[group["Image_Side"] == "Left", new_col] = "Right"
                            group.loc[group["Image_Side"] == "Right", new_col] = "Right"
            return group

        wide["RNFL_severe"] = np.nan
        wide = (
            wide.groupby("RID_Subject")
            .apply(
                compare_sides_severity,
                value_col="Average_RNFL_Thickness(μm)",
                new_col="RNFL_severe",
                smaller=True,
            )
            .reset_index(drop=True)
        )

        wide["HVF_severe"] = np.nan
        wide = (
            wide.groupby("RID_Subject")
            .apply(
                compare_sides_severity,
                value_col="MD",
                new_col="HVF_severe",
                smaller=True,
            )
            .reset_index(drop=True)
        )

        wide["CDR_severe"] = np.nan
        wide = (
            wide.groupby("RID_Subject")
            .apply(
                compare_sides_severity,
                value_col="CDR",
                new_col="CDR_severe",
                smaller=False,
            )
            .reset_index(drop=True)
        )

        def check_severity(row):
            # "Left/Right" and "Right" should return true, and "Left/Right" and "Left" should return true, but "Left" and "Right" should return false
            # old method
            # return row['RNFL_severe'] != row['HVF_severe'] or row['RNFL_severe'] != row['CDR_severe'] or row['HVF_severe'] != row['CDR_severe']
            severities = [row["RNFL_severe"], row["HVF_severe"], row["CDR_severe"]]
            try:
                return not (
                    all(["Left" in l for l in severities])
                    or all(["Right" in l for l in severities])
                )
            except Exception:  # if row is all nan
                return True

        wide["Severity_Mismatch"] = wide.apply(check_severity, axis=1)

        return wide

    def transform_data(self, multimodal_wide, fx_cols, y_method="all_glaucoma"):
        """
        Transforms multimodal data to create X_transformed and y as 0 and 1's; to apply to wide_train and wide_test
        Args:
            - y_method: "all_glaucoma" (Glaucoma=1, GS=0), "urgent_glaucoma" (MD<=-6 AND ICD10 diagnosis code of Glaucoma = 1, else =0), "urgent_glaucoma_nomild" (exclude mild glaucoma as class)
        """

        ##### transform y and drop NA rows #####

        ### drop rows missing label (ie no label for POAG vs PACG vs GS)
        multimodal_wide = multimodal_wide.dropna(subset=["Condition_Label"])
        # drop rows where label is "Other" (should only be PACG, POAG, or GS)
        allowed_labels = ["PACG", "POAG", "GS"]
        multimodal_wide = multimodal_wide[
            multimodal_wide["Condition_Label"].isin(allowed_labels)
        ]

        # combine PACG and POAG as glaucoma
        multimodal_wide["combined_label"] = multimodal_wide["Condition_Label"].replace(
            ["POAG", "PACG"], "Glaucoma"
        )

        ### ICD10 as a fx_col -- need to drop rows missing ICD10 before creating y
        if "ICD10_label_full" in fx_cols:
            # drop rows missing label (ie no label for POAG vs PACG vs GS)
            multimodal_wide = multimodal_wide.dropna(subset=["ICD10_label_full"])
            # drop rows where label is "Other" (should only be PACG, POAG, or GS)
            allowed_labels = ["PACG", "POAG", "GS"]
            multimodal_wide = multimodal_wide[
                multimodal_wide["ICD10_label_full"].isin(allowed_labels)
            ]
    
            # combine PACG and POAG as glaucoma
            multimodal_wide["ICD10_label_full"] = multimodal_wide["ICD10_label_full"].replace(
                ["POAG", "PACG"], "Glaucoma"
            )

        if y_method == "all_glaucoma":
            y = multimodal_wide.combined_label  # Target variable
        elif y_method == "urgent_glaucoma":
            # drop rows missing MD
            multimodal_wide = multimodal_wide.dropna(subset=["MD"])
            multimodal_wide["MD_label"] = multimodal_wide["MD"].apply(
                lambda x: "mod-severe" if x <= -6 else "mild-GS"
            )
            y = multimodal_wide.apply(
                lambda row: (
                    "mod-severe"
                    if (row["combined_label"] == "Glaucoma")
                    and (row["MD_label"] == "mod-severe")
                    else "mild-GS"
                ),
                axis=1,
            )
        elif y_method == "urgent_glaucoma_nomild":
            # drop rows missing MD
            multimodal_wide = multimodal_wide.dropna(subset=["MD"])
            multimodal_wide["MD_label"] = multimodal_wide["MD"].apply(
                lambda x: "mod-severe" if x <= -6 else "mild-GS"
            )
            # drop rows with mild glaucoma
            multimodal_wide = multimodal_wide[
                ~(
                    (multimodal_wide["MD_label"] == "mild-GS")
                    & (multimodal_wide["combined_label"] == "Glaucoma")
                )
            ]
            y = multimodal_wide.apply(
                lambda row: (
                    "mod-severe"
                    if (row["combined_label"] == "Glaucoma")
                    and (row["MD_label"] == "mod-severe")
                    else "GS"
                ),
                axis=1,
            )
        elif y_method == "MD_only":
            # drop rows missing MD
            multimodal_wide = multimodal_wide.dropna(subset=["MD"])
            multimodal_wide["MD_label"] = multimodal_wide["MD"].apply(
                lambda x: "mod-severe" if x <= -6 else "mild-GS"
            )
            y = multimodal_wide.MD_label
        elif y_method == "glaucoma_lowMD":
            # drop all rows with MD>-6
            multimodal_wide = multimodal_wide[multimodal_wide.MD <= -6]
            y = multimodal_wide.combined_label  # Target variable
        elif y_method == "glaucoma_highMD":
            # drop all rows with MD<=-6
            multimodal_wide = multimodal_wide[multimodal_wide.MD > -6]
            y = multimodal_wide.combined_label  # Target variable
        else:
            print("Not a valid y method")

        # convert to 0 and 1
        label_encoder = preprocessing.LabelEncoder()
        y[:] = label_encoder.fit_transform(y)  # fit_transform combines fit and transform
        y = y.astype(int)

        ### transform X ###
        x = multimodal_wide[fx_cols]  # Features


        ### GHT: reformat as "Outside Normal Limits", "Within Normal Limits", "Borderline", "Other"
        if "GHT" in fx_cols:
            ght_categories = [
                "Outside Normal Limits",
                "Within Normal Limits",
                "Borderline",
            ]
            x.loc[~x["GHT"].isin(ght_categories), "GHT"] = (
                np.nan
            )  # alt: 'Other'; I did np.nan bc I feel like it makes more sense to drop this variable

        ### Ethnicity: reformat so that Multi-racial, Other, and ethnicity not specified are combined as Other
        if "Subject_Ethnicity" in fx_cols:
            eth_categories = ["African Descent", "Asian", "Caucasian", "Latin American"]
            x.loc[~x["Subject_Ethnicity"].isin(eth_categories), "Subject_Ethnicity"] = (
                "Other"
            )

        ### categorical data: encode using OneHotEncoder
        categorical_vars = list(
            set(fx_cols) & {"Subject_Gender", "Subject_Ethnicity", "GHT", "ICD10_label_full"}
        )  # cateogorical vars that exist

        if len(categorical_vars) > 0:
            # replace NaN with category "Unknown", then delete this column from one-hot encoding later
            for var in categorical_vars:
                x[var] = x[var].fillna("Unknown")

            encoder = OneHotEncoder(variables=categorical_vars)
            x_transformed = encoder.fit_transform(x)

            # delete Unknown columns
            x_transformed.drop(
                list(x_transformed.filter(regex="Unknown")), axis=1, inplace=True
            )

            # if any columns missing, still keep them and make the column all zeros
            # i did this manually for now for ones I know go missing
            # for var in ["Subject_Ethnicity_African Descent", 'GHT_Borderline']:
            #    if var not in X_transformed.columns:
            #        X_transformed[var]=0

            ### sort categorical encoded columns so that they're in alphabetical order
            def sort_cols(x, var):
                # Select the subset of columns to sort
                subset_columns = [col for col in x.columns if col.startswith(var)]
                # Sort the subset of columns alphabetically
                sorted_columns = sorted(subset_columns)
                # Reorder the DataFrame based on the sorted columns
                sorted_df = x[
                    [col for col in x.columns if col not in subset_columns]
                    + sorted_columns
                ]
                return sorted_df

            for var in categorical_vars:
                x_transformed = sort_cols(x_transformed, var)

        else:
            print("No categorical variables")
            x_transformed = x

        ### format numerical data
        # VFI
        if "VFI" in fx_cols:
            x_transformed["VFI"] = x_transformed["VFI"].replace(
                "Off", np.nan
            )  # replace "Off" with nan

            def convert_percent(x):
                if pd.isnull(x):
                    return np.nan
                return float(x.strip("%")) / 100

            x_transformed["VFI"] = x_transformed["VFI"].map(convert_percent)

        return x_transformed, y

    # current severity rule: prioritize RNFL > HVF > CDR
    # if you don't want thresholds, just make threshold 0
    # just return the first eye if RNFL, MD, CDR all NaN
    def pick_severe_eye(self, df, rnfl_threshold, md_threshold):
        # Sort by 'Average_RNFL_Thickness(μm)', 'MD', and 'CDR' in descending order
        df = df.sort_values(
            by=["Average_RNFL_Thickness(μm)", "MD", "CDR"],
            ascending=[True, True, False],
        )

        ### 1. if only 1 eye has a label, just pick that eye as more severe eye (for Dr. Song's patients)
        df = (
            df.groupby("RID_Subject")
            .apply(lambda group: group[group["Condition_Label"].notna()])
            .reset_index(drop=True)
        )

        # 2. Select the row/eye with most severe value within the thresholds
        def select_row(group):
            max_value = group[
                "Average_RNFL_Thickness(μm)"
            ].min()  # min is more severe for RNFL
            within_value_threshold = group[
                np.abs(group["Average_RNFL_Thickness(μm)"] - max_value)
                <= rnfl_threshold
            ]  # identify eyes within threshold

            if (
                len(within_value_threshold) > 1 or len(within_value_threshold) == 0
            ):  # if both eyes "equal" RNFL OR if RNFL is NaN, then try MD
                max_other_column = within_value_threshold[
                    "MD"
                ].min()  # min is more severe for MD
                within_other_column_threshold = within_value_threshold[
                    np.abs(within_value_threshold["MD"] - max_other_column)
                    <= md_threshold
                ]

                if (
                    len(within_other_column_threshold) > 1
                    or len(within_other_column_threshold) == 0
                ):  # if both eyes "equal" MD OR if MD is NaN, then try CDR
                    return group.sort_values(by=["CDR"], ascending=[False]).iloc[
                        0
                    ]  # since i didn't set CDR threshold, this will always pick something (even if NaN)
                else:
                    return within_other_column_threshold.iloc[0]
            else:
                return within_value_threshold.iloc[0]

        return df.groupby("RID_Subject").apply(select_row).reset_index(drop=True)

    def standardize_data(self, fx_cols, x_train, x_test):
        # identify categorical vs numeric columns
        categorical_vars = ["Subject_Gender", "Subject_Ethnicity", "GHT", "ICD10_label_full"]
        numeric_vars = sorted(set(fx_cols) - set(categorical_vars), key=fx_cols.index)
        if len(numeric_vars)>0:
            scaler = StandardScaler()
    
            # normalize numeric columns for X_train
            normalized_numeric_x_train = pd.DataFrame(
                scaler.fit_transform(x_train[numeric_vars]), columns=numeric_vars
            )
            cat_df = x_train.drop(numeric_vars, axis=1)
            x_train = pd.concat(
                [normalized_numeric_x_train.set_index(cat_df.index), cat_df], axis=1
            )
    
            # normalize numeric columsn for X_test, but using scaler fitted to training data to prevent data leakage
            normalized_numeric_x_test = pd.DataFrame(
                scaler.transform(x_test[numeric_vars]), columns=numeric_vars
            )
            cat_df = x_test.drop(numeric_vars, axis=1)
            x_test = pd.concat(
                [normalized_numeric_x_test.set_index(cat_df.index), cat_df], axis=1
            )

        return x_train, x_test

    # simple imputation fitted to X_train, but also applied to X_test
    def simple_impute(self, x_train_keep_missing, x_test_keep_missing, strat="mean"):
        """
        STRATEGIES
        If “mean”, then replace missing values using the mean along each column. Can only be used with numeric data.

        If “median”, then replace missing values using the median along each column. Can only be used with numeric data.

        If “most_frequent”, then replace missing using the most frequent value along each column. Can be used with strings or numeric data. If there is more than one such value, only the smallest is returned.

        If “constant”, then replace missing values with fill_value. Can be used with strings or numeric data.
        """
        from sklearn.impute import SimpleImputer

        imputer = SimpleImputer(missing_values=np.nan, strategy=strat)
        imputer = imputer.fit(x_train_keep_missing)
        x_train_imputed = imputer.transform(x_train_keep_missing)
        x_test_imputed = imputer.transform(x_test_keep_missing)
        # convert into pandas dataframe instead of np array
        x_train = pd.DataFrame(x_train_imputed, columns=x_train_keep_missing.columns)
        x_test = pd.DataFrame(x_test_imputed, columns=x_test_keep_missing.columns)

        return x_train, x_test

    # return list of pandas dataframes, each containing 1 of 10 imputations
    def mult_impute_missing(self, x, train_data=None):
        if train_data is None:
            train_data = x

        ### multiple imputation method using IterativeImputer from sklearn
        from sklearn.experimental import enable_iterative_imputer
        from sklearn.impute import IterativeImputer

        imp = IterativeImputer(max_iter=10, random_state=0, sample_posterior=True)

        imputed_datasets = []
        for i in range(10):  # 3-10 imputations standard
            imp.random_state = i
            imp.fit(train_data)
            x_imputed = imp.transform(x)
            imputed_datasets.append(pd.DataFrame(x_imputed, columns=x.columns))

        # ALTERNATIVE
        # from statsmodels.imputation import mice.MICEData # alternative package for MICE imputation
        # official docs: https://www.statsmodels.org/dev/generated/statsmodels.imputation.mice.MICE.html#statsmodels.imputation.mice.MICE
        # multiple imputation example using statsmodels: https://github.com/kshedden/mice_workshop
        # imp = mice.MICEData(data)
        # fml = 'y ~ x1 + x2 + x3 + x4' # variables used in multiple imputation model
        # mice = mice.MICE(fml, sm.OLS, imp) # OLS chosen; can change this up
        # results = mice.fit(10, 10) # 10 burn-in cycles to skip, 10 imputations
        # print(results.summary())

        return imputed_datasets

    # Logistic Regression Model Methods###
    ### 2 ways to calculate p-values; NOTE THAT P VALUES MAY NOT MAKE SENSE FOR REGULARIZED MODELS
    # https://stackoverflow.com/questions/25122999/scikit-learn-how-to-check-coefficients-significance
    @staticmethod
    def logit_pvalue(model, x):
        """Calculate z-scores for scikit-learn LogisticRegression.
        parameters:
            model: fitted sklearn.linear_model.LogisticRegression with intercept and large C
            x:     matrix on which the model was fit
        This function uses asymtptics for maximum likelihood estimates.
        """
        p = model.predict_proba(x)
        n = len(p)
        m = len(model.coef_[0]) + 1
        coefs = np.concatenate([model.intercept_, model.coef_[0]])
        x_full = np.matrix(np.insert(np.array(x), 0, 1, axis=1))
        ans = np.zeros((m, m))
        for i in range(n):
            ans = (
                ans
                + np.dot(np.transpose(x_full[i, :]), x_full[i, :]) * p[i, 1] * p[i, 0]
            )
        vcov = np.linalg.inv(np.matrix(ans))
        se = np.sqrt(np.diag(vcov))
        t = coefs / se
        p = (1 - norm.cdf(abs(t))) * 2
        return p

    @staticmethod
    def format_dec(decimals):
        func = np.vectorize(lambda x: "<.001" if x < 0.001 else "%.3f" % x)
        return func(decimals)

    # print model coefficients, ORs, p-values
    def model_summary(self, model, x_train, format_dec=True):
        print("Training set: %i" % len(x_train))
        coefs = model.coef_[0]
        # odd ratios = e^coef
        ors = np.exp(coefs)
        intercept = model.intercept_[0]

        p_values = self.logit_pvalue(model, x_train)

        # compare with statsmodels ### RESULT: produces same result except gives nan instead of 1.00 for insignficant p-values
        # import statsmodels.api as sm
        # sm_model = sm.Logit(y_train.reset_index(drop=True), sm.add_constant(X_train)).fit(disp=0) ### this uses y_train from outside this function so not really valid but oh well I just want it for testing purposes
        # p_values=sm_model.pvalues
        # print(self.format_dec(pvalues))
        # sm_model.summary()

        # print results
        if format_dec:
            results = pd.DataFrame(
                {
                    "Coefficient": np.append(intercept, coefs),
                    "Odds Ratio": self.format_dec(np.append(np.exp(intercept), ors)),
                    "P-value": self.format_dec(p_values),
                },
                index=["Intercept"] + list(x_train.columns),
            )
        else:
            results = pd.DataFrame(
                {
                    "Coefficient": np.append(intercept, coefs),
                    "Odds Ratio": np.append(np.exp(intercept), ors),
                    "P-value": p_values,
                },
                index=["Intercept"] + list(x_train.columns),
            )
        print(results)
        print("")

    # model performance
    # https://medium.com/javarevisited/evaluating-the-logistic-regression-ae2decf42d61
    # helper function for compute_performance(_youden)
    @staticmethod
    def calc_stats(y_pred, y_test):
        # evaluate predictions
        mae = metrics.mean_absolute_error(y_test, y_pred)
        print("MAE: %.3f" % mae)

        # examine the class distribution of the testing set (using a Pandas Series method)
        # y_test.value_counts()
        # calculate the percentage of ones
        # because y_test only contains ones and zeros, we can simply calculate the mean = percentage of ones
        # y_test.mean()
        # calculate the percentage of zeros
        # 1 - y_test.mean()

        # # Metrics computed from a confusion matrix (before thresholding)

        # Confusion matrix is used to evaluate the correctness of a classification model
        cmatrix = confusion_matrix(y_test, y_pred)

        tp = cmatrix[1, 1]
        tn = cmatrix[0, 0]
        fp = cmatrix[0, 1]
        fn = cmatrix[1, 0]

        # Classification Accuracy: Overall, how often is the classifier correct?
        # use float to perform true division, not integer division
        # print((TP + TN) / sum(map(sum, cmatrix))) -- this is the same as the below automatic method
        print("Accuracy: %.3f" % metrics.accuracy_score(y_test, y_pred))

        # Sensitivity(recall): When the actual value is positive, how often is the prediction correct?
        sensitivity = tp / float(fn + tp)
        print("Sensitivity: %.3f" % sensitivity)
        # print('Recall score: %.3f' % metrics.recall_score(y_test, y_pred)) # same thing as sensitivity, but recall term used in ML

        # Specificity: When the actual value is negative, how often is the prediction correct?
        specificity = tn / float(tn + fp)
        print("Specificity: %.3f" % specificity)

        # from imblearn.metrics import specificity_score
        # specificity_score(y_test, y_pred)

        # Precision: When a positive value is predicted, how often is the prediction correct?
        precision = tp / float(tp + fp)
        # print('Precision: %.3f' % precision)
        print("Precision: %.3f" % metrics.precision_score(y_test, y_pred))

        # F score
        f_score = 2 * tp / float(2 * tp + fp + fn)
        # print('F score: %.3f' % f_score)
        print("F1 score: %.3f" % metrics.f1_score(y_test, y_pred))

        # Youden's index: = TPR - FPR = Sensitivity + Specificity - 1
        print(
            "Calculated Youden's J index using predictions: %.3f"
            % (sensitivity + specificity - 1)
        )

        # Evaluate the model using other performance metrics - REDUNDANT, COMMENTED OUT FOR NOW
        # from sklearn.metrics import classification_report
        # print(classification_report(y_test,y_pred))

        # display confusion matrix
        cm_display = metrics.ConfusionMatrixDisplay(
            confusion_matrix=cmatrix, display_labels=None
        )
        cm_display.plot()
        plt.show()

    def compute_performance(self, model, x_test, y_test):
        print("Test set: %i" % len(x_test))
        y_pred = model.predict(x_test)

        print("-------Stats using prediction_probability of 0.5-------")
        self.calc_stats(y_pred, y_test)

    # output performance stats corresponding to OPTIMAL prediction probability cutoff per Youden's, instead of per 0.5 cutoff
    # plot_auc = True: plot individual AUC plot. If False, save to plot onto combined plot later
    def compute_performance_youden(self, model, x_test, y_test, plot=True):
        print("Model features: %s" % x_test.columns.tolist())
        # AUC
        y_pred_proba = model.predict_proba(x_test)[::, 1]
        fpr, tpr, thresholds = metrics.roc_curve(y_test, y_pred_proba)
        auc = metrics.roc_auc_score(y_test, y_pred_proba)
        print("AUC: %.3f" % auc)

        # Youden's J index = sens + spec - 1 = tpr + (1-fpr) -1 = tpr - fpr
        optimal_idx = np.argmax(tpr - fpr)
        optimal_threshold = thresholds[optimal_idx]
        print(
            "Optimal prediction probability threshold by Youdens J index: %.3f"
            % optimal_threshold
        )
        youdens = tpr[optimal_idx] - fpr[optimal_idx]
        print("Optimal Youden's J index: %.3f" % youdens)
        print("Optimal Sensitivity: %.3f" % tpr[optimal_idx])
        print("Optimal Specificity: %.3f" % (1 - fpr[optimal_idx]))

        ### this is not exactly the same as the optimal numbers because it summarizes the data into predictions based on youden's optimal threshold, then computes stats based on those predictions
        # print("-------Stats using prediction_probability per YOUDEN'S-------")
        y_pred = [
            1 if y > optimal_threshold else 0 for y in y_pred_proba
        ]  # Predictions using optimal threshold
        # self.calc_stats(y_pred, y_test)

        if plot:
            # display confusion matrix
            cmatrix = confusion_matrix(y_test, y_pred)
            # display confusion matrix
            cm_display = metrics.ConfusionMatrixDisplay(
                confusion_matrix=cmatrix, display_labels=None
            )
            cm_display.plot()
            plt.show()

            # ROC curve plot with optimal threshold
            plt.plot(fpr, tpr, label="AUC=%.3f, Youden's=%.3f" % (auc, youdens))
            plt.xlabel("False positive rate (1-specificity)")
            plt.ylabel("True positive rate (sensitivity)")
            plt.title("ROC Curve")
            plt.plot([0, 1], [0, 1], "k--")
            plt.scatter(
                fpr[optimal_idx],
                tpr[optimal_idx],
                marker="o",
                color="red",
                label="Optimal Threshold",
            )
            plt.legend(loc=4)
            plt.show()
        return fpr, tpr, auc, optimal_idx, optimal_threshold

    #####Multiple Imputation Logistic Regression analysis methods#####
    ### After performing logistic regression on each imputed dataset, pool the results using Rubin’s rules to obtain a single set of estimates.
    # print model coefficients, ORs, p-values
    def model_summary_mice(self, logreg_models, xtrain_finals):
        print("Training set: %i" % len(xtrain_finals[0]))

        # Extract coefficients and standard errors
        coefs = np.array([model.coef_[0] for model in logreg_models])
        ors = np.exp(coefs)
        intercepts = np.array([model.intercept_[0] for model in logreg_models])
        p_values = np.array(
            [
                self.logit_pvalue(model, xtrain_finals[i])
                for i, model in enumerate(logreg_models)
            ]
        )

        # Calculate pooled estimates
        pooled_coefs = np.mean(coefs, axis=0)
        pooled_ors = np.mean(ors, axis=0)
        pooled_intercept = np.mean(intercepts)
        # I think this calculates SES between the imputed datasets
        pooled_ses = np.sqrt(
            np.mean(coefs**2, axis=0)
            + np.var(coefs, axis=0, ddof=1) * (1 + 1 / len(logreg_models))
        )

        pooled_p_values = np.mean(p_values, axis=0)

        # Display pooled results
        results = pd.DataFrame(
            {
                "Coefficient": np.append(pooled_intercept, pooled_coefs),
                "Odds Ratio": self.format_dec(
                    np.append(np.exp(pooled_intercept), pooled_ors)
                ),
                "Standard Error": self.format_dec(
                    np.append(np.nan, pooled_ses)
                ),  # Intercept SE is not calculated here
                "P-value": self.format_dec(pooled_p_values),
            },
            index=["Intercept"] + list(xtrain_finals[0].columns),
        )
        print(results)
        print("")

    # model performance
    # https://medium.com/javarevisited/evaluating-the-logistic-regression-ae2decf42d61
    def compute_performance_mice(self, logreg_models, xtest_finals, y_test):
        print("Test set: %i" % len(xtest_finals[0]))

        y_pred_results = []
        y_pred_proba_results = []
        for model, x_test in zip(logreg_models, xtest_finals):
            y_pred_results.append(model.predict(x_test))

            y_pred_proba_results.append(model.predict_proba(x_test)[::, 1])

        ypred_df = pd.DataFrame(np.row_stack(y_pred_results))
        y_pred = np.array(
            ypred_df.mode(axis=0).loc[0].astype(int)
        )  ##### used the mode of y_pred across the 10 imputations
        y_pred_proba = np.mean(y_pred_proba_results, axis=0)

        # evaluate predictions
        mae = metrics.mean_absolute_error(y_test, y_pred)
        print("MAE: %.3f" % mae)

        # examine the class distribution of the testing set (using a Pandas Series method)
        y_test.value_counts()

        # calculate the percentage of ones
        # because y_test only contains ones and zeros, we can simply calculate the mean = percentage of ones
        y_test.mean()

        # calculate the percentage of zeros
        1 - y_test.mean()

        # # Metrics computed from a confusion matrix (before thresholding)

        # Confusion matrix is used to evaluate the correctness of a classification model
        cmatrix = confusion_matrix(y_test, y_pred)

        tp = cmatrix[1, 1]
        tn = cmatrix[0, 0]
        fp = cmatrix[0, 1]
        fn = cmatrix[1, 0]

        # Classification Accuracy: Overall, how often is the classifier correct?
        # use float to perform true division, not integer division
        # print((TP + TN) / sum(map(sum, confusion_matrix))) -- this is is the same as the below automatic method
        print("Accuracy: %.3f" % metrics.accuracy_score(y_test, y_pred))

        # Sensitivity(recall): When the actual value is positive, how often is the prediction correct?
        sensitivity = tp / float(fn + tp)

        print("Sensitivity: %.3f" % sensitivity)
        # print('Recall score: %.3f' % metrics.recall_score(y_test, y_pred)) # same thing as sensitivity, but recall term used in ML

        # Specificity: When the actual value is negative, how often is the prediction correct?
        specificity = tn / float(tn + fp)
        print("Specificity: %.3f" % specificity)

        # from imblearn.metrics import specificity_score
        # specificity_score(y_test, y_pred)

        # Precision: When a positive value is predicted, how often is the prediction correct?
        precision = tp / float(tp + fp)
        # print('Precision: %.3f' % precision)
        print("Precision: %.3f" % metrics.precision_score(y_test, y_pred))

        # F score
        f_score = 2 * tp / float(2 * tp + fp + fn)
        # print('F score: %.3f' % f_score)
        print("F1 score: %.3f" % metrics.f1_score(y_test, y_pred))

        # Evaluate the model using other performance metrics - REDUNDANT, COMMENTED OUT FOR NOW
        # from sklearn.metrics import classification_report
        # print(classification_report(y_test,y_pred))

        # AUC
        fpr, tpr, _ = metrics.roc_curve(y_test, y_pred_proba)
        auc = metrics.roc_auc_score(y_test, y_pred_proba)
        print("AUC: %.3f" % auc)

        # CM matrix plot
        from sklearn import metrics

        cm_display = metrics.ConfusionMatrixDisplay(
            confusion_matrix=confusion_matrix, display_labels=None
        )

        cm_display.plot()
        plt.show()
        # 0 = GS, 1 = POAG

        # ROC curve plot
        plt.plot(fpr, tpr, label="auc=" + str(auc))
        plt.xlabel("False positive rate (1-specificity)")
        plt.ylabel("True positive rate (sensitivity)")
        plt.legend(loc=4)
        plt.show()
