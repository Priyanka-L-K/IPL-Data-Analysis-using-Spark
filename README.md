# IPL-Data-Analysis-using-Spark


from pyspark.sql import SparkSession
from pyspark.sql.functions import *
from abc import ABC, abstractmethod

# Abstract Factory
class IPLDataAnalysisFactory(ABC):
    @abstractmethod
    def create_loader(self):
        pass
    
    @abstractmethod
    def create_transformer(self):
        pass
    
    @abstractmethod
    def create_analyzer(self):
        pass

# Abstract Products
class DataLoader(ABC):
    @abstractmethod
    def load_data(self, spark):
        pass

class DataTransformer(ABC):
    @abstractmethod
    def transform_data(self, df):
        pass

class DataAnalyzer(ABC):
    @abstractmethod
    def analyze_data(self, df):
        pass

# Concrete Factory
class IPL2022DataAnalysisFactory(IPLDataAnalysisFactory):
    def create_loader(self):
        return IPL2022DataLoader()
    
    def create_transformer(self):
        return IPL2022DataTransformer()
    
    def create_analyzer(self):
        return IPL2022DataAnalyzer()

# Concrete Products
class IPL2022DataLoader(DataLoader):
    def load_data(self, spark):
        return spark.read.csv("IPL_2022.csv", header=True, inferSchema=True)

class IPL2022DataTransformer(DataTransformer):
    def transform_data(self, df):
        df = df.withColumn("match_date", to_date(col("match_date"), "dd-MM-yyyy"))
        return df

class IPL2022DataAnalyzer(DataAnalyzer):
    def analyze_data(self, df):
        # Top run scorers
        top_run_scorers = df.groupBy("batter").agg(sum("batsman_run").alias("total_runs")).orderBy(desc("total_runs")).limit(10)
        top_run_scorers.show()

        # Top wicket takers
        top_wicket_takers = df.filter(col("kind").isNotNull()).groupBy("bowler").count().withColumnRenamed("count", "wickets").orderBy(desc("wickets")).limit(10)
        top_wicket_takers.show()

        # Matches won by each team
        matches_won = df.filter(col("innings") == 2).groupBy("winner").count().withColumnRenamed("count", "matches_won").orderBy(desc("matches_won"))
        matches_won.show()

        # Top 10 players with most Player of the Match awards
        player_of_match = df.filter(col("innings") == 2).groupBy("player_of_match").count().withColumnRenamed("count", "awards").orderBy(desc("awards")).limit(10)
        player_of_match.show()

# Main application
def run_ipl_analysis(factory):
    spark = SparkSession.builder.appName("IPL 2022 Data Analysis").getOrCreate()

    loader = factory.create_loader()
    transformer = factory.create_transformer()
    analyzer = factory.create_analyzer()

    df = loader.load_data(spark)
    transformed_df = transformer.transform_data(df)
    analyzer.analyze_data(transformed_df)

    spark.stop()

if __name__ == "__main__":
    factory = IPL2022DataAnalysisFactory()
    run_ipl_analysis(factory)
