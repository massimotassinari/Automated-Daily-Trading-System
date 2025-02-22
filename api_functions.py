# Installation: pip install simfin
# import simfin package
import simfin as sf

# Set your API-key for downloading data.
sf.set_api_key('344dd533-861f-4bef-9f52-be02f0276014')

# Set the local directory where data-files are stored.
# The directory will be created if it does not already exist.
sf.set_data_dir('data/simfin_data/')

# Download the data from the SimFin server and load into a Pandas DataFrame.
df = sf.load_shareprices(variant='latest')

# Print the first rows of the data.
print(df.head())


# Installation: pip install simfin
# import simfin package
import simfin as sf

# Set your API-key for downloading data.
sf.set_api_key('344dd533-861f-4bef-9f52-be02f0276014')

# Set the local directory where data-files are stored.
# The directory will be created if it does not already exist.
sf.set_data_dir('data/simfin_data/')

# Download the data from the SimFin server and load into a Pandas DataFrame.
df = sf.load_companies(market='us')

# Print the first rows of the data.
print(df.head())