import numpy as np
import pandas as pd
import xgboost as xgb
from itertools import compress
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_selection import chi2
from sklearn.metrics import log_loss
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder


def transform_review_scores(df, resp='review_scores_rating'):
    """Keep observations with review scores and transform them to five levels
    
    :param df: data frame to be transformed
    :param resp: response
    :return: transformed data frame
    """
    df = df[~df[resp].isnull()].reset_index(0, True)
    df[resp] = pd.cut(df[resp], np.arange(0, 120, 20), labels=np.arange(5), include_lowest=True)
    return df


def transform_zipcode(df):
    """Transform zipcode to outward code
    
    :param df: data frame to be transformed
    :return: transformed data frame
    """
    variable = 'zipcode'
    df[variable] = df[variable].fillna('')
    df[variable] = df[variable].str.upper().str.replace('[^0-9A-Z]+', '')
    df.loc[df[variable].str.len() >= 8, variable] = ''  # Remove wrongly recorded zipcode
    df.loc[df[variable].str.len() >= 5, variable] = df.loc[df[variable].str.len() >= 5, variable] \
        .str.slice(0, -3)  # Extract outward code
    return df


def transform_label(df, variables):
    """Encode labels using numerical values
    
    :param df: data frame to be transformed
    :param variables: list of variables to be encoded
    :return: transformed data frame
    """
    encoder = LabelEncoder()
    for variable in variables:
        df[variable] = df[variable].fillna('')
        df[variable] = encoder.fit_transform(df[variable])
    return df


def transform_host_since(df):
    """Transform host start date to duration
    
    :param df: data frame to be transformed
    :return: transformed data frame
    """
    df['host_since'] = pd.to_datetime(df['host_since'], yearfirst=True)
    df['host_for'] = (df['host_since'].max() - df['host_since']).dt.days
    return df


def transform_boolean(df, variables):
    """Transform boolean values
    
    :param df: data frame to be transformed
    :param variables: list of variables to be transformed
    :return: transformed data frame
    """
    for variable in variables:
        df[variable] = df[variable].map({'f': 0, 't': 1}, 'ignore')
    return df


def transform_text(df, variables, max_features=100):
    """Transform variables consisting of text into a sparse format
    
    :param df: data frame to be transformed
    :param variables: list of variables to be transformed
    :param max_features: maximum number of features
    :return: transformed data frame
    """
    vectorizer = CountVectorizer(stop_words='english', max_features=max_features)
    tmp = []
    for variable in variables:
        df[variable] = df[variable].fillna('')
        sparse = vectorizer.fit_transform(df[variable])
        sparse = pd.DataFrame(sparse.toarray(), columns=sorted(vectorizer.vocabulary_.keys())) \
            .add_prefix(variable + '.')  # Rename the extracted features
        tmp.append(sparse)
    df = pd.DataFrame(pd.concat([df] + tmp, 1))
    return df


def transform_percent(df, variables):
    """Transform strings of percentages to decimals
    
    :param df: data frame to be transformed
    :param variables: list of variables to be transformed
    :return: transformed data frame
    """
    for variable in variables:
        df[variable] = df[variable].str.strip('%')
        df[variable] = df[variable].astype(np.float64) / 100
    return df


def transform_price(df, variables):
    """Transform strings of price to numerals
    
    :param df: data framed to be transformed
    :param variables: list of variables to be transformed
    :return: transformed data frame
    """
    for variable in variables:
        df[variable] = df[variable].str.strip('$').str.replace(',', '')  # Remove dollar signs and thousands separators
        df[variable] = df[variable].astype(np.float64)
    return df


def transform_amenities(df):
    """Extract amenities
    
    :param df: data frame to be transformed
    :return: transformed data frame
    """
    variable = 'amenities'
    df[variable] = df[variable].str.replace(r'[:\-\./ ]', '_').str.replace(r'[\(\)]', '').str.replace(r'_+', '_') \
        .str.lower()  # Clean the format of amenities
    df = transform_text(df, [variable])
    columns_to_remove = [column for column in df.columns if 'missing' in column]
    df = pd.DataFrame(df.drop(columns_to_remove, 1))  # Remove unwanted amenities
    return df


def remove_redundant_features(df, variables, resp='review_scores_rating'):
    """Perform chi-squared test to remove sparse features of low significance
    
    :param df: data frame to be transformed
    :param variables: list of variables that have been transformed into a sparse format to be transformed
    :param resp: response
    :return: transformed data frame
    """
    for variable in variables:
        variables_list = [item for item in df.columns if variable + '.' in item]  # Find sparse features
        tmp = df[variables_list + [resp]].dropna()
        _, p_val = chi2(tmp[variables_list], tmp[resp])
        variables_list = list(compress(variables_list, (p_val > 0.05)))  # Find sparse features of low significance
        df = df.drop(variables_list, 1)
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
    df = transform_label(df, ['experiences_offered',
                              'host_response_time',
                              'zipcode',
                              'property_type',
                              'room_type',
                              'bed_type',
                              'cancellation_policy'])
    df = transform_host_since(df)
    df = transform_boolean(df, ['host_is_superhost',
                                'host_has_profile_pic',
                                'host_identity_verified',
                                'is_location_exact',
                                'requires_license',
                                'instant_bookable',
                                'require_guest_profile_picture',
                                'require_guest_phone_verification'])
    df = transform_text(df, ['host_verifications'])
    df = transform_percent(df, ['host_response_rate'])
    df = transform_price(df, ['price'])
    df = transform_amenities(df)
    df = remove_redundant_features(df, ['host_verifications', 'amenities'], resp)

    df = df.drop(['listing_id',
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
                  'host_verifications',
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
                  'amenities',
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
                  'cancellation_policy',
                  'require_guest_profile_picture',
                  'require_guest_phone_verification',
                  'reviews_per_month'], 1)

    return df


def transform(df):
    """Transform the data frame and split it into the train and test sets
    
    :param df: data frame (listings) to be transformed
    :return: split train and test sets
    """
    resp = 'review_scores_rating'

    df = transform_listings(df, resp)

    train, test = train_test_split(df)

    x_train = train.drop(resp, 1)
    y_train = train[resp]
    x_test = test.drop(resp, 1)
    y_test = test[resp]

    return x_train, y_train, x_test, y_test


def main():
    listings = pd.read_csv('../src/data_sets/listings.csv.gz')

    x_train, y_train, x_test, y_test = transform(listings)

    params = {'silent': 1,
              'eta': 0.05,
              'max_depth': 5,
              'max_delta_step': 1,
              'subsample': 0.75,
              'colsample_bytree': 0.75,
              'objective': 'multi:softprob',
              'num_class': 5,
              'eval_metric': 'mlogloss'}
    d_train = xgb.DMatrix(x_train, y_train)
    num_boost_round = 100

    bst = xgb.train(params, d_train, num_boost_round)

    d_test = xgb.DMatrix(x_test)
    pred = bst.predict(d_test)

    m_log_loss = log_loss(y_test, pred, labels=np.arange(5))

    print(m_log_loss)


if __name__ == '__main__':
    main()
