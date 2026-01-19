from sklearn.preprocessing import OneHotEncoder
data = [
    ['pizza'],
    ['burger'],
    ['pasta'],
    ['pizza']
]

encoder = OneHotEncoder()
encoded = encoder.fit_transform(data).toarray()

print("categories:", encoder.categories_)
print("Encoded output:")
for row in encoded:
    print(row)