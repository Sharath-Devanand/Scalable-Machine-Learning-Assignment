from pyspark.sql import SparkSession
from pyspark.sql.functions import split,regexp_extract
import matplotlib.pyplot as plt
from pyspark.sql.functions import split
import seaborn as sns
import os


spark = SparkSession.builder \
    .master("local[2]") \
    .appName("COM6012 Spark Intro") \
    .config("spark.local.dir", os.environ.get('TMPDIR', '/tmp')) \
    .getOrCreate()

sc = spark.sparkContext
sc.setLogLevel("WARN")  # This can only affect the log level after it is executed.

data_path = "Data/NASA_access_log_Jul95.gz"
logFile = spark.read.text(data_path)


# Define schema for log data
log_schema = "host STRING, timestamp STRING, request STRING, status INT, content_size INT"

# Split each line into columns according to the schema
log_columns = split(logFile.value, ' ')

# Create DataFrame with the columns
log_df = logFile.withColumn("host", log_columns[0]) \
                .withColumn("timestamp", log_columns[3] + " " + log_columns[4]) \
                .withColumn("request", log_columns[5]) \
                .withColumn("status", log_columns[8].cast("int")) \
                .withColumn("content_size", log_columns[9].cast("int"))

hostsGermany = log_df.filter(log_df["host"].rlike("\\.de$")).count()
hostsCanada = log_df.filter(log_df["host"].rlike("\\.ca$")).count()
hostsSingapore = log_df.filter(log_df["host"].rlike("\\.sg$")).count()

#Plot results as bar graph

print("Number of total hosts from Germany: ", hostsGermany)
print("Number of total hosts from Canada: ", hostsCanada)
print("Number of total hosts from Singapore: ", hostsSingapore)

countries = ['Germany', 'Canada', 'Singapore']
hosts = [hostsGermany, hostsCanada, hostsSingapore]

plt.bar(countries, hosts)

plt.xlabel('Countries')
plt.ylabel('Number of hosts')
plt.title('Number of hosts from different countries')
plt.show()

plt.savefig('Q1_figure.png')

"""
For each of the three countries in Question A (Germany, Canada, and Singapore), find the number
of unique hosts and the top 9 most frequent hosts for each country. You need to write the three numbers and 3
x 9 = 27 hosts in total into a text file.
"""

unique_hosts_canada = log_df.filter(log_df["host"].rlike("\\.ca$")).select("host").distinct().count()
unique_hosts_germany = log_df.filter(log_df["host"].rlike("\\.de$")).select("host").distinct().count()
unique_hosts_singapore = log_df.filter(log_df["host"].rlike("\\.sg$")).select("host").distinct().count()

# Top hosts with count

top_hosts_germany = log_df.filter(log_df["host"].rlike("\\.de$")).groupBy("host").count().sort("count", ascending=False).limit(9).collect()
top_hosts_canada = log_df.filter(log_df["host"].rlike("\\.ca$")).groupBy("host").count().sort("count", ascending=False).limit(9).collect()
top_hosts_singapore = log_df.filter(log_df["host"].rlike("\\.sg$")).groupBy("host").count().sort("count", ascending=False).limit(9).collect()

## Print the results
print("Number of unqiue hosts in Germany: ", unique_hosts_germany)
print("\n\nNumber of unique hosts in Canada: ", unique_hosts_canada)
print("\n\nNUmber of unique hosts in Singapore: ", unique_hosts_singapore)

print("\n\nTop hosts from Germany:")
for host in top_hosts_germany:
    print(host['host'], host['count'])

print("\n\nTop hosts from Canada:")
for host in top_hosts_canada:
    print(host['host'], host['count'])

print("\n\nTop hosts from Singapore:")
for host in top_hosts_singapore:
    print(host['host'], host['count'])


# For each country, visualise the percentage (with respect to the total in that country) of requests by
# each of the top 9 most frequent hosts and the rest (i.e. 10 proportions in total) using a graph of your
# choice with the 9 hosts clearly labelled on the graph. Three graphs need to be produced.

#Germany


# Calculate the count of the rest
total_count = log_df.filter(log_df["host"].rlike("\\.de$")).count()
rest_count = total_count - sum(host['count'] for host in top_hosts_germany)

# Create a list of hosts and counts for the pie chart
labels = [host['host'] for host in top_hosts_germany] + ['Rest']
counts = [host['count'] for host in top_hosts_germany] + [rest_count]

# Plot the pie chart
plt.figure(figsize=(8, 8))
plt.pie(counts, labels=labels, autopct='%1.1f%%', startangle=140)
plt.title('Top 9 Hosts from Germany')
plt.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.

# Save the pie chart to a PNG file
plt.savefig('Q1_figure2.png')


##Canada

# Calculate the count of the rest
total_count = log_df.filter(log_df["host"].rlike("\\.ca$")).count()
rest_count = total_count - sum(host['count'] for host in top_hosts_canada)

# Create a list of hosts and counts for the pie chart
labels = [host['host'] for host in top_hosts_canada] + ['Rest']
counts = [host['count'] for host in top_hosts_canada] + [rest_count]

# Plot the pie chart
plt.figure(figsize=(8, 8))
plt.pie(counts, labels=labels, autopct='%1.1f%%', startangle=140)
plt.title('Top 9 Hosts from Canada')
plt.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.

# Save the pie chart to a PNG file
plt.savefig('Q1_figure3.png')


#Singapore

# Calculate the count of the rest
total_count = log_df.filter(log_df["host"].rlike("\\.sg$")).count()
rest_count = total_count - sum(host['count'] for host in top_hosts_singapore)

# Create a list of hosts and counts for the pie chart
labels = [host['host'] for host in top_hosts_singapore] + ['Rest']
counts = [host['count'] for host in top_hosts_singapore] + [rest_count]

# Plot the pie chart
plt.figure(figsize=(8, 8))
plt.pie(counts, labels=labels, autopct='%1.1f%%', startangle=140)
plt.title('Top 9 Hosts from Singapore')
plt.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.

# Save the pie chart to a PNG file
plt.savefig('Q1_figure4.png')


"""
For the most frequent host from each of the three countries, produce a heatmap plot with day as
the x-axis (the range of x-axis should cover the range of days available in the log file. If there are 31 days,
it runs from 1st to 31st. If it starts from 5th and ends on 25th, it runs from 5th to 25th), the hour of visit as
the y-axis (0 to 23, as recorded on the server), and the number of visits indicated by the colour. Three x-y
heatmap plots need to be produced with the day and hour clearly labelled.
"""


# Define a regular expression pattern to extract day, hour, and host
pattern = r'^(\S+) - - \[(\d{2})/(\w{3})/(\d{4}):(\d{2}):(\d{2}):(\d{2})'

# Extract day, hour, and host using regexp_extract
log_df = logFile.withColumn("host", regexp_extract("value", pattern, 1)) \
    .withColumn("day", regexp_extract("value", pattern, 2)) \
    .withColumn("hour", regexp_extract("value", pattern, 5))
               



# Filter log data for hosts ending with ".de"
log_df_germany = log_df.filter(log_df["host"].rlike("\\.de$"))

# Aggregate the number of visits for each day and hour combination
heatmap_data = log_df_germany.groupBy("day", "hour").count().toPandas()
heatmap_data = heatmap_data.sort_values(by="hour", ascending=True)

# Create a heatmap plot
plt.figure(figsize=(12, 8))
sns.heatmap(heatmap_data.pivot(index="hour", columns="day", values="count"), cmap="YlGnBu", linewidths=1,square=True,cbar=True)
plt.title("Heatmap of Visits from German Hosts")
plt.xlabel("Day")
plt.ylabel("Hour of Visit")

plt.savefig('Q1_figure5.png')

# Filter log data for hosts ending with ".ca"

log_df_canada = log_df.filter(log_df["host"].rlike("\\.ca$"))

# Aggregate the number of visits for each day and hour combination
heatmap_data = log_df_canada.groupBy("day", "hour").count().toPandas()
heatmap_data = heatmap_data.sort_values(by="hour", ascending=True)

# Create a heatmap plot
plt.figure(figsize=(12, 8))
sns.heatmap(heatmap_data.pivot(index="hour", columns="day", values="count"), cmap="YlGnBu", linewidths=1,square=True,cbar=True)
plt.title("Heatmap of Visits from Canadian Hosts")
plt.xlabel("Day")
plt.ylabel("Hour of Visit")

plt.savefig('Q1_figure6.png')

# Filter log data for hosts ending with ".sg"

log_df_singapore = log_df.filter(log_df["host"].rlike("\\.sg$"))

# Aggregate the number of visits for each day and hour combination
heatmap_data = log_df_singapore.groupBy("day", "hour").count().toPandas()
heatmap_data = heatmap_data.sort_values(by="hour", ascending=True)

# Create a heatmap plot
plt.figure(figsize=(12, 8))
sns.heatmap(heatmap_data.pivot(index="hour", columns="day", values="count"), cmap="YlGnBu", linewidths=1,square=True,cbar=True)
plt.title("Heatmap of Visits from Singaporean Hosts")
plt.xlabel("Day")
plt.ylabel("Hour of Visit")

plt.savefig('Q1_figure7.png')

spark.stop()