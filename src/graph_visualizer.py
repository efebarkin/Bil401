import networkx as nx
import matplotlib.pyplot as plt

class GraphVisualizer:
    def __init__(self, spark):
        self.spark = spark

    def create_graph(self, ratings_df, movies_df):
        # GraphFrames ile graf oluştur
        # Görselleştirme kodları buraya gelecek
        pass

    def visualize(self, G):
        plt.figure(figsize=(12, 8))
        nx.draw(G, with_labels=True)
        plt.show()