import numpy as np
import pandas as pd
import xgboost as xgb
from itertools import compress
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_selection import chi2
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder


def transform_zipcode(df):
    variable = 'zipcode'
    df[variable] = df[variable].fillna('')
    df[variable] = df[variable].str.upper().str.replace('[^0-9A-Z]+', '')
    df.loc[df[variable].str.len() >= 8, variable] = ''
    df.loc[df[variable].str.len() >= 5, variable] = df.loc[df[variable].str.len() >= 5, variable].str.slice(0, -3)
    return df


def transform_label(df, variables):
    encoder = LabelEncoder()
    for variable in variables:
        df[variable] = df[variable].fillna('')
        df[variable] = encoder.fit_transform(df[variable])
    return df


def transform_host_since(df):
    df['host_since'] = pd.to_datetime(df['host_since'], yearfirst=True)
    df['host_for'] = (pd.Timestamp.now() - df['host_since']).dt.days
    return df


def transform_boolean(df, variables):
    for variable in variables:
        df[variable] = df[variable].map({'f': 0, 't': 1}, 'ignore')
    return df


def transform_text(df, variables):
    vectorizer = CountVectorizer(stop_words='english', max_features=100)
    tmp = []
    for variable in variables:
        df[variable] = df[variable].fillna('')
        sparse = vectorizer.fit_transform(df[variable])
        sparse = pd.DataFrame(sparse.toarray(), columns=sorted(vectorizer.vocabulary_.keys())) \
            .add_prefix(variable + '.')
        tmp.append(sparse)
    df = pd.DataFrame(pd.concat([df] + tmp, 1))
    return df


def transform_percent(df, variables):
    for variable in variables:
        df[variable] = df[variable].str.strip('%')
        df[variable] = df[variable].astype(np.float64) / 100
    return df


def transform_price(df, variables):
    for variable in variables:
        df[variable] = df[variable].str.strip('$').str.replace(',', '')
        df[variable] = df[variable].astype(np.float64)
    return df


def transform_amenities(df):
    variable = 'amenities'
    df[variable] = df[variable].str.replace(r'[:\-\./ ]', '_').str.replace(r'[\(\)]', '').str.replace(r'_+', '_') \
        .str.lower()
    df = transform_text(df, [variable])
    columns_to_remove = [column for column in df.columns if 'missing' in column]
    df = pd.DataFrame(df.drop(columns_to_remove, 1))
    return df


def remove_redundant_features(df, variables, resp):
    for variable in variables:
        variables_list = [item for item in df.columns if variable + '.' in item]
        tmp = df[variables_list + [resp]].dropna()
        _, p_val = chi2(tmp[variables_list], tmp[resp])
        variables_list = list(compress(variables_list, (p_val > 0.05)))
        df = df.drop(variables_list, 1)
    return df


def transform_listings(df, resp):
    df = df.rename(columns={'id': 'listing_id'})
    df = df[~df[resp].isnull()].reset_index(0, True)
    df[resp] = pd.cut(df[resp], np.arange(0, 120, 20), labels=np.arange(5), include_lowest=True)

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


def compute_lambda(count, k=5, f=5):
    return 1 / (1 + np.exp(-(count - k) / f))


def compute_prob(row):
    prior, posterior, lambda_val = row
    return lambda_val * posterior + (1 - lambda_val) * prior


def reformat(num, name, resp, variable):
    num = num.pivot(index=variable, columns=resp, values=variable + '.' + name).reset_index().fillna(0)
    columns = dict(zip(np.arange(5), [variable + '.' + name + '.' + score for score in np.arange(5).astype(str)]))
    num = num.rename(columns=columns)
    return num


def add_noise(df, tmp, variable, r=0.01):
    df = df.merge(tmp, 'left').drop(variable, 1)
    noise = 1 + np.random.uniform(-0.5, 0.5, (len(df), 5)) * r
    df[[variable + '.prob.' + str(i) for i in np.arange(5)]] *= noise
    return df


def transform_high_cardinality(train, test, resp, variables):
    prior = train[resp].value_counts(True).sort_index().reset_index()
    prior['index'] = prior['index'].map(lambda x: 'prior.' + str(x))
    prior = prior.transpose()
    prior = prior.rename(columns=prior.iloc[0]).drop('index', 0).reset_index()

    for variable in variables:
        new_prior = pd.DataFrame(pd.concat([prior] * len(train[variable].unique()), ignore_index=True))
        new_prior['index'] = train[variable].unique()
        new_prior = new_prior.rename(columns={'index': variable})

        posterior = train.groupby(variable)[resp].value_counts(True).rename(variable + '.posterior').reset_index()
        posterior = reformat(posterior, 'posterior', resp, variable)

        count = train.groupby([variable, resp])[resp].count().rename(variable + '.count').reset_index()
        count[resp] = count[resp].astype(np.int64)
        count = reformat(count, 'count', resp, variable)

        tmp = new_prior.merge(posterior).merge(count)

        for i in np.arange(5):
            tmp[variable + '.lambda.' + str(i)] = tmp[variable + '.count.' + str(i)].map(compute_lambda)
            tmp[variable + '.prob.' + str(i)] = tmp[['prior.' + str(i),
                                                     variable + '.posterior.' + str(i),
                                                     variable + '.lambda.' + str(i)]].apply(compute_prob, 1)

        tmp = tmp[[variable] + [variable + '.prob.' + str(i) for i in np.arange(5)]]

        train = add_noise(train, tmp, variable)
        test = add_noise(test, tmp, variable)

    return train, test


def transform(df):
    resp = 'review_scores_rating'

    df = transform_listings(df, resp)

    train, test = train_test_split(df)
    train = train.reset_index(0, True)
    test = test.reset_index(0, True)

    train, test = transform_high_cardinality(train, test, resp, ['host_id', 'zipcode'])

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
              'subsample': 0.75,
              'colsample_bytree': 0.75,
              'objective': 'multi:softmax',
              'num_class': 5,
              'eval_metric': 'mlogloss'}
    d_train = xgb.DMatrix(x_train, y_train)
    num_boost_round = 1000

    bst = xgb.train(params, d_train, num_boost_round)

    d_test = xgb.DMatrix(x_test)
    pred = bst.predict(d_test)

    print(pred)


if __name__ == '__main__':
    main()
