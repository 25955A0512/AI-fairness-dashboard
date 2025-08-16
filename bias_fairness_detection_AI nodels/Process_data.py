from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

# Encode categorical features
for col in ["workclass", "education", "marital-status", "occupation", "relationship", "race", "sex", "native-country", "income"]:
    data[col] = LabelEncoder().fit_transform(data[col])

# Define features and target
X = data.drop("income", axis=1)
y = data["income"]
sensitive_feature = data["sex"]  # 0 = Female, 1 = Male

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
