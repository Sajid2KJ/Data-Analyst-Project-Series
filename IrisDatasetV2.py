import seaborn as sns
import matplotlib.pyplot as plt
df = sns.load_dataset('iris')

df.hist(bins=15, figsize=(10, 10))
plt.suptitle('Histograms of Iris Features')
plt.show()

for feature in df.columns[:-1]:
    sns.kdeplot(df[feature], shade=True)
    plt.title(f'Distribution of {feature}')
    plt.xlabel(feature)
    plt.show()

plt.figure(figsize=(10, 8))
df.boxplot()
plt.title('Boxplot of Iris Features')
plt.show()

sns.pairplot(df, hue='species', diag_kind='kde')
plt.suptitle('Pairwise Relationships in Iris Dataset', y=1.02)
plt.show()

correlation_matrix = df.corr()
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', linewidths=0.5)
plt.title('Correlation Matrix of Iris Features')
plt.show()

for feature in df.columns[:-1]:
    sns.violinplot(x='species', y=feature, data=df)
    plt.title(f'Violin Plot of {feature} by Species')
    plt.show()

for feature in df.columns[:-1]:
    sns.boxplot(x='species', y=feature, data=df)
    plt.title(f'Box Plot of {feature} by Species')
    plt.show()