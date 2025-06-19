import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans


def load_data(csv_path: str) -> pd.DataFrame:
    """Load the Pokemon CSV dataset."""
    return pd.read_csv(csv_path)


def visualize_skills(df: pd.DataFrame) -> None:
    """Visualize the distribution of Pokemon skills."""
    skills = ["HP", "Attack", "Defense", "Sp. Atk", "Sp. Def", "Speed"]
    df_melted = df[skills].melt(var_name="Skill", value_name="Value")
    sns.boxplot(x="Skill", y="Value", data=df_melted)
    plt.title("Pokemon Skills Distribution")
    plt.show()


def run_kmeans(df: pd.DataFrame, n_clusters: int = 5):
    """Run KMeans on standardised Pokemon skills."""
    skills = ["HP", "Attack", "Defense", "Sp. Atk", "Sp. Def", "Speed"]
    scaler = StandardScaler()
    features = scaler.fit_transform(df[skills])
    kmeans = KMeans(n_clusters=n_clusters, random_state=0)
    df["Cluster"] = kmeans.fit_predict(features)
    return df, kmeans


def plot_clusters(df: pd.DataFrame) -> None:
    """Scatter plot of Attack vs Defense colored by cluster."""
    sns.scatterplot(x="Attack", y="Defense", hue="Cluster", data=df, palette="Set2")
    plt.title("Clusters of Pokemon")
    plt.show()


def main() -> None:
    df = load_data("Pokemon.csv")
    visualize_skills(df)
    df, _ = run_kmeans(df, n_clusters=5)
    plot_clusters(df)


if __name__ == "__main__":
    main()
