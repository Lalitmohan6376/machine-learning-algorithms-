from sklearn.preprocessing import LabelEncoder

grades = ['A','B','C','D','E']
encoder = LabelEncoder()
encoded_values = encoder.fit_transform(grades)
print(encoded_values)