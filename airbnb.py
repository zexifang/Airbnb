import ast
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import xgboost as xgb


# Set variables for weighting the prior and posterior
K = 5
F = 5
R = 0.01


def vectorize(df, variables):
    """Extract text features
    
    :param df: data frame
    :param variables: list of variables to be extracted
    :return: a list of data frames of extracted text features
    """
    vectorizer = CountVectorizer(stop_words='english', max_features=100)
    tmp = []
    for variable in variables:
        df[variable] = df[variable].fillna('')
        sparse = vectorizer.fit_transform(df[variable])
        sparse = pd.DataFrame(sparse.toarray(), columns=sorted(vectorizer.vocabulary_.keys())) \
            .add_prefix(variable + '_')
        tmp.append(sparse)
    return tmp


def encode(df, variables):
    """Encode labels
    
    :param df: data frame
    :param variables: list of variables to be encoded
    :return: encoded data frame
    """
    encoder = LabelEncoder()
    for variable in variables:
        df[variable] = df[variable].fillna('')
        df[variable] = encoder.fit_transform(df[variable])
    return df


def transform_amenities(df):
    """Extract text features of amenities
    
    :param df: data frame
    :return: transformed data frame
    """
    variable = 'amenities'
    df[variable] = df[variable].str.replace(' / ', '_').str.replace('-', '_').str.replace('.', '_') \
        .str.replace('/', '_').str.replace(' ', '_').str.replace(':', '')  # Reformat amenities
    tmp = vectorize(df, [variable])
    df = pd.DataFrame(pd.concat([df] + tmp, 1))
    columns_to_remove = [column for column in df.columns if 'missing' in column]
    df = pd.DataFrame(df.drop(columns_to_remove, 1))  # Delete unwanted amenities
    return df


def transform_boolean(df, variables):
    """Transform boolean values
    
    :param df: data frame
    :param variables: list of variables to be transformed
    :return: transformed data frame
    """
    for variable in variables:
        df[variable] = df[variable].map({'f': 0, 't': 1}, 'ignore')
    return df


def transform_counts(df, variables):
    """Transform strings to lists and count the number of elements
    
    :param df: data frame
    :param variables: list of variables to be transformed
    :return: transformed data frame
    """
    for variable in variables:
        df[variable] = df[variable].map(lambda x: ast.literal_eval(x))
        df[variable] = df[variable].map(len, 'ignore')
    return df


def transform_host_since(df):
    """Transform host start date to duration
    
    :param df: data frame
    :return: transformed data frame
    """
    df['host_since'] = pd.to_datetime(df['host_since'], yearfirst=True)
    df['host_for'] = (pd.Timestamp.now() - df['host_since']).dt.days
    return df


def transform_percent(df, variables):
    """Transform strings of percentage to decimals
    
    :param df: data frame
    :param variables: list of variables
    :return: transformed data frame
    """
    for variable in variables:
        df[variable] = df[variable].map(lambda x: float(x.strip('%')) / 100, 'ignore')
    return df


def transform_price(df, variables):
    """Transform strings of price to numerals
    
    :param df: data frame
    :param variables: list of variables
    :return: transformed data frame
    """
    for variable in variables:
        df[variable] = df[variable].map(lambda x: float(x.strip('$').replace(',', '')), 'ignore')
    # Calculate unit price
    df['price_per_bathroom'] = df['price'] / df['bathrooms']
    df['price_per_bedroom'] = df['price'] / df['bedrooms']
    df['price_per_bed'] = df['price'] / df['beds']
    df['price_per_square_foot'] = df['price'] / df['square_feet']
    df['price_per_person'] = df['price'] / df['accommodates']
    return df


def transform_review_scores_rating(df):
    """Transform review scores to 10 levels
    
    :param df: data frame
    :return: transformed data frame
    """
    df['review_scores_rating'] = pd.cut(df['review_scores_rating'], range(0, 101, 10), labels=range(1, 11)) \
        .astype(int) - 1
    return df


def transform_calendar(df):
    """Transform calendar data
    
    :param df: data frame
    :return: transformed data frame
    """
    df = transform_boolean(df, ['available'])
    # Calculate average availability
    df = df.groupby('listing_id')['available'].mean().reset_index().rename(columns={'available': 'availability'})
    return df


def transform_listings(df):
    """Transform listings data
    
    :param df: data frame
    :return: transformed data frame
    """
    df = df[pd.notnull(df['review_scores_rating'])].reset_index()  # Remove null review scores
    df = df.rename(columns={'id': 'listing_id'})
    df = pd.DataFrame(pd.concat([df] + vectorize(df, ['summary', 'house_rules', 'host_about']), 1))
    df = encode(df, ['listing_id', 'experiences_offered', 'host_id', 'host_response_time', 'neighbourhood_cleansed',
                     'zipcode', 'property_type', 'room_type', 'bed_type', 'cancellation_policy'])
    df = transform_host_since(df)
    df = transform_boolean(df, ['host_is_superhost', 'host_has_profile_pic', 'host_identity_verified',
                                'is_location_exact', 'requires_license', 'instant_bookable',
                                'require_guest_profile_picture', 'require_guest_phone_verification'])
    df = transform_counts(df, ['host_verifications'])
    df = transform_percent(df, ['host_response_rate'])
    df = transform_price(df, ['price', 'weekly_price', 'monthly_price', 'security_deposit', 'cleaning_fee',
                              'extra_people'])
    df = transform_amenities(df)
    df = transform_review_scores_rating(df)
    df = df.drop(['listing_url', 'scrape_id', 'last_scraped', 'name', 'summary', 'space', 'description',
                  'neighborhood_overview', 'notes', 'transit', 'access', 'interaction', 'house_rules', 'thumbnail_url',
                  'medium_url', 'picture_url', 'xl_picture_url', 'host_url', 'host_name', 'host_since', 'host_location',
                  'host_about', 'host_thumbnail_url', 'host_picture_url', 'host_neighbourhood', 'street',
                  'neighbourhood', 'neighbourhood_group_cleansed', 'city', 'state', 'market', 'smart_location',
                  'country_code', 'country', 'latitude', 'longitude', 'amenities', 'calendar_updated',
                  'has_availability', 'availability_30', 'availability_60', 'availability_90', 'availability_365',
                  'calendar_last_scraped', 'first_review', 'last_review', 'review_scores_accuracy',
                  'review_scores_cleanliness', 'review_scores_checkin', 'review_scores_communication',
                  'review_scores_location', 'review_scores_value', 'license', 'jurisdiction_names'], 1)
    return df


def transform_reviews(df):
    """Transform reviews data
    
    :param df: data frame
    :return: transformed data frame
    """
    df['comments'] = df['comments'].fillna('')
    comments = df.groupby('listing_id')['comments'].apply(lambda x: ' '.join(x)).reset_index()  # Combine comments
    number_of_reviewers = df.groupby('listing_id')['reviewer_id'].nunique().reset_index() \
        .rename(columns={'reviewer_id': 'number_of_reviewers'})  # Count the number of unique reviewers
    df = pd.DataFrame(pd.concat([number_of_reviewers] + vectorize(comments, ['comments']), 1))
    return df


def compute_lambda(count):
    """Compute the weight of the prior and posterior
    
    :param count: sample size
    :return: weight
    """
    return 1 / (1 + np.exp(-(count - K) / F))


def transform_train_test(train, test):
    """Transform high-cardinality categorical variables to adjusted probability
    
    :param train: train set
    :param test: test set
    :return: train set and test set
    """
    # Calculate the prior
    prior = train['review_scores_rating'].map(train['review_scores_rating'].value_counts(True)).rename('prior')
    train = pd.DataFrame(pd.concat([train, prior], 1))

    variables = ['host_id', 'zipcode']
    for variable in variables:
        posterior = train.groupby(variable)['review_scores_rating'].value_counts(True) \
            .rename(variable + '_' + 'posterior').reset_index()  # Calculate the posterior
        count = train.groupby([variable, 'review_scores_rating'])['review_scores_rating'].count() \
            .rename(variable + '_' + 'count').reset_index()
        train = train.merge(posterior, 'left', [variable, 'review_scores_rating']) \
            .merge(count, 'left', [variable, 'review_scores_rating'])
        train[variable + '_' + 'lambda'] = train[variable + '_' + 'count'].map(compute_lambda)
        train[variable + '_' + 'prob'] = train[variable + '_' + 'lambda'] * train[variable + '_' + 'posterior'] \
            + (1 - train[variable + '_' + 'lambda']) * train['prior']  # Calculate the adjusted probability
        test = test.merge(train[[variable, variable + '_' + 'prob']], 'left', variable)
        train[variable + '_' + 'prob'] = train[variable + '_' + 'prob'] \
            * (1 + np.random.uniform(-0.5, 0.5, len(train)) * R)
        test[variable + '_' + 'prob'] = test[variable + '_' + 'prob'] \
            * (1 + np.random.uniform(-0.5, 0.5, len(test)) * R)

    train = pd.DataFrame(train.drop(['prior'] + [variable + '_posterior' for variable in variables]
                                    + [variable + '_count' for variable in variables]
                                    + [variable + '_lambda' for variable in variables], 1))

    return train, test


def transform(calendar, listings, reviews):
    """Transform data
    
    :param calendar: calendar data
    :param listings: listings data
    :param reviews: reviews data
    :return: 
    """
    calendar = transform_calendar(calendar)
    listings = transform_listings(listings)
    reviews = transform_reviews(reviews)

    df = listings.merge(calendar, 'left', 'listing_id').merge(reviews, 'left', 'listing_id')  # Merge transformed data

    train, test = train_test_split(df)  # Split train set and test set
    train, test = transform_train_test(train, test)  # Transform train set and test set

    target = 'review_scores_rating'

    x_train = train.drop(target, 1)
    y_train = train[target]
    x_test = test.drop(target, 1)
    y_test = test[target]

    return x_train, y_train, x_test, y_test


def fit(x_train, y_train):
    """Train the booster
    
    :param x_train: predictors
    :param y_train: response
    :return: trained booster
    """
    d_train = xgb.DMatrix(x_train, y_train)

    params = {'silent': 1, 'eta': 0.05, 'max_depth': 5, 'subsample': 0.75, 'colsample_bytree': 0.75,
              'objective': 'multi:softprob', 'num_class': 10, 'eval_metric': 'mlogloss'}
    num_boost_round = 1000

    bst = xgb.train(params, d_train, num_boost_round)

    return bst


def predict(x_test, bst):
    """Predict review scores levels
    
    :param x_test: predictors
    :param bst: trained booster
    :return: prediction
    """
    d_test = xgb.DMatrix(x_test)
    pred = bst.predict(d_test)

    return pred


def main():
    calendar = pd.read_csv('../src/data_sets/calendar.csv.gz')
    listings = pd.read_csv('../src/data_sets/listings.csv.gz')
    reviews = pd.read_csv('../src/data_sets/reviews.csv.gz')

    x_train, y_train, x_test, y_test = transform(calendar, listings, reviews)

    bst = fit(x_train, y_train)
    pred = predict(x_test, bst)

    print(pred)

if __name__ == '__main__':
    main()
