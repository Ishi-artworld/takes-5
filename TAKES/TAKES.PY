import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
df = pd.read_csv("train.csv")
df.head()

df.info()
df.describe()
df['Survived'].value_counts()
df['Sex'].value_counts()
df['Pclass'].value_counts()
plt.hist(df['Age'].dropna(), bins=30)
plt.title("Age Distribution")
plt.xlabel("Age")
plt.ylabel("Frequency")
plt.show()
sns.boxplot(x=df['Fare'])
plt.title("Fare Distribution")
plt.show()
sns.countplot(x='Sex', hue='Survived', data=df)
plt.title("Survival by Gender")
plt.show()
sns.countplot(x='Pclass', hue='Survived', data=df)
plt.title("Survival by Class")
plt.show()
sns.scatterplot(x='Age', y='Fare', hue='Survived', data=df)
plt.title("Age vs Fare")
plt.show()
sns.pairplot(df[['Survived','Age','Fare','Pclass']], hue='Survived')
plt.show()
corr = df.corr(numeric_only=True)

sns.heatmap(corr, annot=True, cmap='coolwarm')
plt.title("Correlation Heatmap")
plt.show()

















