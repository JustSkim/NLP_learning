from sklearn.preprocessing import OneHotEncoder

enc = OneHotEncoder()
X = [['male', 'from US', 'uses Safari'], ['female', 'from Europe', 'uses Firefox']]
enc.fit(X)
print(enc.categories_)
"""[array(['female', 'male'], dtype=object), array(['from Europe', 'from US'], dtype=object), array(['uses Firefox', 'uses Safari'], dtype=object)]"""
print(enc.get_feature_names())
"""['x0_female' 'x0_male' 'x1_from Europe' 'x1_from US' 'x2_uses Firefox' 'x2_uses Safari']"""
print(enc.transform([['female', 'from US', 'uses Safari'],
                     ['male', 'from Europe', 'uses Safari']]).toarray())
"""
[[1. 0. 0. 1. 0. 1.]
 [0. 1. 1. 0. 0. 1.]]
"""
from sklearn.feature_extraction.text import CountVectorizer

