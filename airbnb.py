"""Main module.

This module is the prediction of the review scores of Airbnb data.
"""


import numpy as np
import pandas as pd

from itertools import compress
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_selection import chi2
from sklearn.metrics import log_loss
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer
from sklearn.preprocessing import Imputer
from sklearn.utils import resample


def transform_review_scores(df, resp='review_scores_rating'):
    """Keep observations with review scores and transform them to five levels
    
    :param df: data frame to be transformed
    :param resp: response
    :return: transformed data frame
    """
    df = df.copy()
    df = df[~df[resp].isnull()].reset_index(0, True)  # Remove missing values
    df[resp] = pd.cut(df[resp], np.arange(0, 120, 20), labels=np.arange(5), include_lowest=True) \
        # Bin response into five levels
    return df


def transform_zipcode(df, variable='zipcode'):
    """Transform zipcode to outward code
    
    :param df: data frame to be transformed
    :param variable: zipcode
    :return: transformed data frame
    """
    df = df.copy()
    df[variable] = df[variable].fillna('').str.lower().str.replace(r' +', '')  # Reformat zipcode
    pat = df[variable].str.match(r'^[a-z]{1,2}[0-9][a-z0-9]?([0-9][a-z]{2})?$')  # Find zipcode of correct format
    df.loc[~pat, variable] = ''  # Remove wrongly recorded zipcode
    df[variable] = df[variable].str.extract(r'([a-z]{1,2})', expand=False)  # Extract postcode area
    return df


def transform_label(df, variables=None):
    """Encode labels using numerical values
    
    :param df: data frame to be transformed
    :param variables: list of variables to be encoded
    :return: transformed data frame
    """
    df = df.copy()
    if variables is None:
        variables = []

    label_binarizer = LabelBinarizer()
    transformed_list = []

    for variable in variables:
        df[variable] = df[variable].fillna('')  # Reformat variables
        transformed = label_binarizer.fit_transform(df[variable])  # Binarize variables

        # Set column names of the transformed data
        if transformed.shape[1] == 1:
            transformed_columns = [variable]
        else:
            columns = pd.Series(label_binarizer.classes_).str.lower().str.replace(r'[&\-/ ]', '_') \
                .str.replace(r'_+', '_')
            transformed_columns = [variable + '.' + column for column in columns]

        transformed = pd.DataFrame(transformed, columns=transformed_columns)
        transformed_list.append(transformed)  # Add transformed data to the transformed list
        df = df.drop(variable, 1)  # Remove original data

    df = pd.DataFrame(pd.concat([df] + transformed_list, 1))  # Concatenate transformed data
    return df


def transform_host_since(df):
    """Transform host start date to duration
    
    :param df: data frame to be transformed
    :return: transformed data frame
    """
    df = df.copy()
    df['host_since'] = pd.to_datetime(df['host_since'], yearfirst=True)  # Convert to datetime format
    df['host_for'] = (df['host_since'].max() - df['host_since']).dt.days  # Calculate the duration of hosting
    return df


def transform_text(df, variables=None, max_features=100):
    """Transform variables consisting of text into a sparse format
    
    :param df: data frame to be transformed
    :param variables: list of variables to be transformed
    :param max_features: maximum number of features
    :return: transformed data frame
    """
    df = df.copy()
    if variables is None:
        variables = []

    count_vectorizer = CountVectorizer(stop_words='english', max_features=max_features)
    transformed_list = []

    for variable in variables:
        df[variable] = df[variable].fillna('')
        transformed = count_vectorizer.fit_transform(df[variable])

        # Set column names of the transformed data
        columns = sorted(count_vectorizer.vocabulary_.keys())
        transformed_columns = [variable + '.' + column for column in columns]

        transformed = pd.DataFrame(transformed.toarray(), columns=transformed_columns)
        transformed_list.append(transformed)  # Add transformed data to the transformed list
        df = df.drop(variable, 1)  # Remove original data

    df = pd.DataFrame(pd.concat([df] + transformed_list, 1))  # Concatenate transformed data
    return df


def transform_percent(df, variables=None):
    """Transform strings of percentages to decimals
    
    :param df: data frame to be transformed
    :param variables: list of variables to be transformed
    :return: transformed data frame
    """
    df = df.copy()
    if variables is None:
        variables = []
    for variable in variables:
        df[variable] = df[variable].str.strip('%')
        df[variable] = df[variable].astype(np.float64) / 100
    return df


def transform_price(df, variables=None):
    """Transform strings of price to numerals
    
    :param df: data framed to be transformed
    :param variables: list of variables to be transformed
    :return: transformed data frame
    """
    df = df.copy()
    if variables is None:
        variables = []
    for variable in variables:
        df[variable] = df[variable].str.strip('$').str.replace(',', '')  # Remove dollar signs and thousands separators
        df[variable] = df[variable].astype(np.float64)
    return df


def transform_amenities(df, variable='amenities'):
    """Extract amenities
    
    :param df: data frame to be transformed
    :param variable: amenities
    :return: transformed data frame
    """
    df = df.copy()
    df[variable] = df[variable].str.replace(r'[:\-\./ ]', '_').str.replace(r'[\(\)]', '').str.replace(r'_+', '_') \
        .str.lower()  # Reformat amenities
    df = transform_text(df, [variable])  # Transform amenities using transform_text
    columns_to_remove = [column for column in df.columns if 'missing' in column]
    df = pd.DataFrame(df.drop(columns_to_remove, 1))  # Remove unwanted amenities
    return df


def impute(df):
    """Impute missing values

    :param df: data frame to be imputed
    :return: imputed data frame
    """
    df = df.copy()
    imputer = Imputer(strategy='median')  # Replace missing values with the median
    columns = df.columns
    df = imputer.fit_transform(df)
    df = pd.DataFrame(df, columns=columns)
    return df


def transform_listings(df, resp='review_scores_rating'):
    """Transform listings data
    
    :param df: data frame (listings data) to be transformed
    :param resp: response
    :return: transformed data frame
    """
    df = df.rename(columns={'id': 'listing_id'})

    df = transform_review_scores(df, resp)

    df = transform_zipcode(df)

    vars_label = [
        'experiences_offered',
        'host_response_time',
        'zipcode',
        'property_type',
        'room_type',
        'bed_type',
        'cancellation_policy',
        'host_is_superhost',
        'is_location_exact',
        'requires_license',
        'instant_bookable',
        'require_guest_profile_picture',
        'require_guest_phone_verification'
    ]
    df = transform_label(df, vars_label)

    df = transform_host_since(df)

    df = transform_text(df, ['host_verifications'])

    df = transform_percent(df, ['host_response_rate'])

    df = transform_price(df, ['price'])

    df = transform_amenities(df)

    vars_drop = [
        'listing_id',
        'listing_url',
        'scrape_id',
        'last_scraped',
        'name',
        'summary',
        'space',
        'description',
        'neighborhood_overview',
        'notes',
        'transit',
        'access',
        'interaction',
        'house_rules',
        'thumbnail_url',
        'medium_url',
        'picture_url',
        'xl_picture_url',
        'host_url',
        'host_name',
        'host_since',
        'host_location',
        'host_about',
        'host_acceptance_rate',
        'host_thumbnail_url',
        'host_picture_url',
        'host_neighbourhood',
        'host_listings_count',
        'host_total_listings_count',
        'host_has_profile_pic',
        'host_identity_verified',
        'street',
        'neighbourhood',
        'neighbourhood_cleansed',
        'neighbourhood_group_cleansed',
        'city',
        'state',
        'market',
        'smart_location',
        'country_code',
        'country',
        'latitude',
        'longitude',
        'is_location_exact',
        'square_feet',
        'weekly_price',
        'monthly_price',
        'security_deposit',
        'cleaning_fee',
        'guests_included',
        'extra_people',
        'minimum_nights',
        'maximum_nights',
        'calendar_updated',
        'has_availability',
        'availability_30',
        'availability_60',
        'availability_90',
        'availability_365',
        'calendar_last_scraped',
        'first_review',
        'last_review',
        'review_scores_accuracy',
        'review_scores_cleanliness',
        'review_scores_checkin',
        'review_scores_communication',
        'review_scores_location',
        'review_scores_value',
        'requires_license',
        'license',
        'jurisdiction_names',
        'instant_bookable',
        'require_guest_profile_picture',
        'require_guest_phone_verification',
        'reviews_per_month'
    ]
    df = pd.DataFrame(df.drop(vars_drop, 1))

    df = impute(df)

    return df


def remove_redundant_features(train, test, variables=None, resp='review_scores_rating'):
    """Perform chi-squared test to remove sparse features of low significance

    :param train: train data frame to be transformed
    :param test: test data frame to be transformed
    :param variables: list of ``sparse`` variables to be transformed
    :param resp: response
    :return: transformed data frame
    """
    train = train.copy()
    test = test.copy()
    if variables is None:
        variables = []

    for variable in variables:
        variables_list = [item for item in train.columns if variable + '.' in item]  # Find sparse features
        tmp = train[variables_list + [resp]]  # Form data frame of sparse feature and response
        _, p_val = chi2(tmp[variables_list], tmp[resp])  # Perform chi-squared test
        variables_list = list(compress(variables_list, (p_val > 0.05)))  # Find sparse features of low significance
        train = train.drop(variables_list, 1)  # Remove sparse features of low significance
        test = test.drop(variables_list, 1)  # Remove sparse features of low significance
    return train, test


def upsample(df, resp='review_scores_rating'):
    df = df.copy()
    df_upsampled_list = []
    max_n_samples = df[resp].value_counts().max()  # Find the class with maximum number of samples
    for i in np.arange(5):
        upsampled = resample(df[df[resp] == i], n_samples=max_n_samples)  # Upsample
        df_upsampled_list.append(upsampled)  # Add upsampled data to the transformed list
    df = pd.DataFrame(pd.concat(df_upsampled_list, ignore_index=True))  # Concatenate upsampled data
    return df


def transform(df):
    """Transform the data frame and split it into the train and test sets
    
    :param df: data frame (listings) to be transformed
    :return: split train and test sets
    """
    resp = 'review_scores_rating'

    df = transform_listings(df, resp)  # Perform feature engineering to listings data

    train, test = train_test_split(df)  # Split transformed data into train and test sets

    train, test = remove_redundant_features(train, test, ['host_verifications', 'amenities']) \
        # Remove redundant features using the train set

    train = upsample(train)  # Upsample imbalanced classes

    # Split train and test sets into features and response
    x_train = train.drop(resp, 1)
    y_train = train[resp]
    x_test = test.drop(resp, 1)
    y_test = test[resp]

    return x_train, y_train, x_test, y_test


def fit_and_predict(x_train, y_train, x_test, y_test):
    """Fit the random forest classifier and predict the review scores

    :param x_train: train set features
    :param y_train: train set response
    :param x_test: test set features
    :param y_test: test set response
    :return: logarithmic loss of the test set
    """
    clf = RandomForestClassifier(500)
    clf.fit(x_train, y_train)
    y_pred = clf.predict_proba(x_test)
    m_log_loss = log_loss(y_test, y_pred)
    return m_log_loss


def main():
    listings = pd.read_csv('../src/data_sets/listings.csv.gz')  # Read data

    x_train, y_train, x_test, y_test = transform(listings)  # Transform data

    m_log_loss = fit_and_predict(x_train, y_train, x_test, y_test)  # fit and predict

    print('The logarithmic loss is {}.'.format(m_log_loss))


if __name__ == '__main__':
    main()
