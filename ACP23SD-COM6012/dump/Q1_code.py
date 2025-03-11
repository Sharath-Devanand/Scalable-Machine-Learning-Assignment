from pyspark.sql import SparkSession
import pyspark.sql.functions as F
import matplotlib.pyplot as plt
from pyspark.sql.functions import dayofmonth, hour, count
import seaborn as sns
import numpy as np
import pandas as pd

spark = (
    SparkSession.builder.master("local[2]")
    .appName("COM6012 Spark Intro")
    .config("spark.local.dir", "/mnt/parscratch/users/acr23nm")
    .getOrCreate()
)

sc = spark.sparkContext
sc.setLogLevel("WARN")  # This can only affect the log level after it is executed.

# Load the data

logFile = spark.read.text("./Data/NASA_access_log_Jul95.gz").cache()

# Covert the data into a DataFrame for easier manipulation
data = (
    logFile.withColumn("host", F.regexp_extract("value", "^(.*) - -.*", 1))
    .withColumn("timestamp", F.regexp_extract("value", ".* - - \[(.*)\].*", 1))
    .withColumn("request", F.regexp_extract("value", '.*"(.*)".*', 1))
    .withColumn(
        "HTTP reply code",
        F.split("value", " ").getItem(F.size(F.split("value", " ")) - 2).cast("int"),
    )
    .withColumn(
        "bytes in the reply",
        F.split("value", " ").getItem(F.size(F.split("value", " ")) - 1).cast("int"),
    )
    .drop("value")
    .cache()
)

# A. Total number of requests for 1. all hosts from Germany (.de) 2. all hosts from Canada (.ca) 3. all hosts from Singapore (.sg)
hosts_de = data.filter(data.host.rlike("\.de$")).count()
hosts_ca = data.filter(data.host.rlike("\.ca$")).count()
hosts_sg = data.filter(data.host.rlike("\.sg$")).count()

print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Question A ~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
print(f"The total number of requests from Germany (.de) is: {hosts_de}")
print(f"The total number of requests from Canada (.ca) is: {hosts_ca}")
print(f"The total number of requests from Singapore (.sg) is: {hosts_sg}")
print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")

# Plot graph for total number of requests from Germany (.de), Canada (.ca) and Singapore (.sg)

labels = ["Germany (.de)", "Canada (.ca)", "Singapore (.sg)"]
requests = [(hosts_de), (hosts_ca), (hosts_sg)]

plt.bar(labels, requests, color="blue")
plt.xlabel("Countries")
plt.ylabel("Number of Requests")
plt.title("Total Number of Requests from 3 Countries")
plt.savefig("./Output/Q1_figA.png")


# B. Unique hosts and top 9 most frequent hosts from Germany (.de), Canada (.ca) and Singapore (.sg)
unique_hosts_de = (
    data.select("host").filter(data.host.rlike("\.de$")).distinct().count()
)
unique_hosts_ca = (
    data.select("host").filter(data.host.rlike("\.ca$")).distinct().count()
)
unique_hosts_sg = (
    data.select("host").filter(data.host.rlike("\.sg$")).distinct().count()
)

# Top 9 most frequent hosts from Germany (.de), Canada (.ca) and Singapore (.sg)
freq_hosts_de = (
    data.filter(data.host.rlike("\.de$"))
    .distinct()
    .groupBy("host")
    .count()
    .sort("count", ascending=False)
    .limit(9)
)


freq_hosts_ca = (
    data.filter(data.host.rlike("\.ca$"))
    .distinct()
    .groupBy("host")
    .count()
    .sort("count", ascending=False)
    .limit(9)
)


freq_hosts_sg = (
    data.filter(data.host.rlike("\.sg$"))
    .distinct()
    .groupBy("host")
    .count()
    .sort("count", ascending=False)
    .limit(9)
)

print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Question B ~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
print(f"There are {unique_hosts_de} unique hosts from Germany (.de)")
print(f"There are {unique_hosts_ca} unique hosts from Canada (.ca)")
print(f"There are {unique_hosts_sg} unique hosts from Singapore (.sg)")
print("The top 9 most frequent hosts from Germany (.de) are:" + "\n")
freq_hosts_de.select("host").show(9, False)
print("The top 9 most frequent hosts from Canada (.ca) are:" + "\n")
freq_hosts_ca.select("host").show(9, False)
print("The top 9 most frequent hosts from Singapore (.sg) are:" + "\n")
freq_hosts_sg.select("host").show(9, False)
print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")

# C. For each country, visualise the percentage (with respect to the total in that country) of requests by each of the top 9 most frequent hosts and the rest (i.e. 10 proportions in total) using a graph of your choice with the 9 hosts clearly labelled on the graph. Three graphs need to be produced.

# def plot_request_distribution(freq_hosts_df, total_hosts, country, pic_num):

#     # Calculate sum of top 9 hosts and others
#     top_hosts_sum = freq_hosts_df.groupBy().sum().collect()[0][0]
#     others = total_hosts - top_hosts_sum

#     # Get host names and counts for plotting
#     hosts = freq_hosts_df.rdd.map(lambda row: row[0]).collect()
#     counts = freq_hosts_df.rdd.map(lambda row: row[1]).collect() + [others]
#     labels = hosts + ["Others"]

#     # Plotting
#     plt.figure(figsize=(10, 6))
#     plt.pie(counts, labels=labels, autopct='%1.1f%%', startangle=90)
#     plt.title(f'Requests Distribution for {country}')
#     plt.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
#     plt.savefig(f"./Output/Q1_figC{pic_num}.png", bbox_inches='tight')
#     plt.show()


# This function will create a pie chart for the given frequency data
def plot_request_distribution(freq_hosts_df, total_hosts, country, pic_num):
    # Calculate the sum of the counts for the top 9 hosts
    top_hosts_sum = freq_hosts_df.select("count").rdd.flatMap(lambda x: x).sum()
    others = total_hosts - top_hosts_sum
    
    # Extract host names and counts
    hosts_counts = freq_hosts_df.rdd.map(lambda row: (row['host'], row['count'])).collect()
    hosts, counts = zip(*hosts_counts)
    hosts = list(hosts) + ['Others']
    counts = list(counts) + [others]
    
    # Explode the first slice (the largest one)
    explode = [0.1 if i == counts.index(max(counts)) else 0 for i in range(len(counts))]
    
    # Plotting the pie chart
    fig, ax = plt.subplots()
    wedges, texts, autotexts = ax.pie(counts, explode=explode, labels=hosts, autopct='%1.1f%%',
                                      shadow=True, startangle=90, textprops={'fontsize': 8})
    
    # Draw a circle at the center to turn it into a donut chart
    centre_circle = plt.Circle((0,0),0.70,fc='white')
    fig = plt.gcf()
    fig.gca().add_artist(centre_circle)
    
    # Equal aspect ratio ensures that pie is drawn as a circle
    ax.axis('equal')  
    
    # Add a legend outside of the pie chart
    plt.legend(wedges, hosts, title="Hosts", loc="center left", bbox_to_anchor=(1, 0.5))
    
    plt.title(f"Requests Distribution for {country}")
    plt.savefig(f"./Output/Q1_figC{pic_num}.png", bbox_inches='tight')  # Save the figure with a tight layout
    plt.show()


plot_request_distribution(freq_hosts_de, hosts_de, "Germany (.de)", 1)
plot_request_distribution(freq_hosts_ca, hosts_ca, "Canada (.ca)", 2)
plot_request_distribution(freq_hosts_sg, hosts_sg, "Singapore (.sg)", 3)


# D. For the most frequent host from each of the three countries, produce a heatmap plot with day as the x-axis (the range of x-axis should cover the range of days available in the log file. If there are 31 days, it runs from 1st to 31st. If it starts from 5th and ends on 25th, it runs from 5th to 25th), the hour of visit as the y-axis (0 to 23, as recorded on the server), and the number of visits indicated by the colour. Three x-y heatmap plots need to be produced with the day and hour clearly labelled.

topHost_de = freq_hosts_de.select(freq_hosts_de.host.cast('string').alias('host')).collect()[0]['host']
topHost_ca = freq_hosts_ca.select(freq_hosts_ca.host.cast('string').alias('host')).collect()[0]['host']
topHost_sg = freq_hosts_sg.select(freq_hosts_sg.host.cast('string').alias('host')).collect()[0]['host']

def plot_heatmap(data, host, country, pic_num):
    # Prepare data for heatmap
    day_pattern = r'^(\d{2})/.*'
    hour_pattern = r'^\d{2}/[a-zA-Z]{3}/\d{4}:(\d{2}):.*'
    heatmap_data = data.filter(data.host == host)\
                        .withColumn('day', F.regexp_extract('timestamp', day_pattern, 1).cast('int'))\
                        .withColumn('hour', F.regexp_extract('timestamp', hour_pattern, 1).cast('int'))\
                        .groupBy('day', 'hour')\
                        .agg(F.count('*').alias('count'))
    

    if heatmap_data.count() == 0:
        print(f"No data available for host {host} in {country}.")
        return

    # Convert to Pandas DataFrame for plotting
    heatmap_pandas = heatmap_data.toPandas().pivot_table(index='day', columns='hour', values='count')

    # Ensure the DataFrame covers all hours 0-23 and the days found in logs
    if heatmap_pandas is not None and not heatmap_pandas.empty:
        all_hours = np.arange(24)
        heatmap_pandas = heatmap_pandas.reindex(columns=all_hours, fill_value=0)

        # Safely get min and max day values
        day_min = heatmap_data.select("day").agg({"day": "min"}).collect()[0][0] or 1
        day_max = heatmap_data.select("day").agg({"day": "max"}).collect()[0][0] or 31
        all_days = np.arange(day_min, day_max + 1)
        heatmap_pandas = heatmap_pandas.reindex(index=all_days, fill_value=0)
        heatmap_pandas = heatmap_pandas.fillna(0)
    else:
        print(f"No entries found for {country}.")
        return

    # Plotting
    plt.figure(figsize=(12, 8))
    sns.heatmap(heatmap_pandas, cmap="crest", annot=True, fmt="g", linewidths=.5)
    plt.title(f"Visits Heatmap for {host} ({country})")
    plt.ylabel("Hour of Day")
    plt.xlabel("Day of Month")
    # plt.yticks(np.arange(len(all_hours)), all_hours)
    plt.xticks(np.arange(len(all_days)), all_days)
    plt.savefig(f"./Output/Q1_figD{pic_num}.png")
    plt.show()

# Extract the most frequent host from each country's DataFrame
top_host_de = freq_hosts_de.head(1)[0]['host']
top_host_ca = freq_hosts_ca.head(1)[0]['host']
top_host_sg = freq_hosts_sg.head(1)[0]['host']

plot_heatmap(data, top_host_de, "Germany", 1)
plot_heatmap(data, top_host_ca, "Canada", 2)
plot_heatmap(data, top_host_sg, "Singapore", 3)

sc.stop()