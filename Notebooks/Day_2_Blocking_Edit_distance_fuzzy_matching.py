# data import and manipulation
import pandas as pd
import recordlinkage as rl

# Read-in files with cleaned inventors (PatentsView) and principal investigators (FedRePORTER) names
inventors = pd.read_csv('../Data/inventors_cleaned.csv')
pi = pd.read_csv('../Data/pi_cleaned.csv')
#
# View first rows of the table
inventors.head()
#
# View first rows of the table
pi.head()
#
# Read-in files with cleaned organization names in PatentsView (assignees) and FedRePORTER (organizations)
assignees = pd.read_csv('../Data/assignees_cleaned.csv')
organizations = pd.read_csv('../Data/organizations_cleaned.csv')
#
# View first rows of the table
assignees.head()
#
# View first rows of the table
organizations.head()
# markdown
# ## Record Linkage
#
# The `recordlinkage` package is a quite powerful tool for you to use when you want to link records within a dataset or across multiple datasets. We've already done some pre-processing and then tried deterministic matching.
#
# However, as we have seen in the previous notebook, we might want to consider how strict we want our matching to be. For example, we want to make sure that we catch any typos or common misspellings, but we want to avoid relaxing the matching condition to the point that anything will match anything.
# markdown
# ### Indexing
#
# Indexing allows us to create candidate links, which basically means identifying pairs of data rows which might refer to the same real world entity. This is also called the comparison space (matrix). There are different ways to index data. The easiest is to create a full index and consider every pair a match. This is also the least efficient method, because we will be comparing every row of one dataset with every row of the other dataset.
#
# If we had 10,000 records in data frame A and 100,000 records in data frame B, we would have 1,000,000,000 candidate links. You can see that comparing over a full index is getting inefficient when working with big data.
# markdown
# We can do better if we actually include our knowledge about the data to eliminate bad link from the start. This can be done through blocking. The `recordlinkage` package gives you multiple options for this. For example, you can block by using variables, which means only links exactly equal on specified values will be kept.
# markdown
# Here we will start by blocking on `city` and `state`, to narrow down the number of candidate links.
# markdown
# You can try and see how the number of candidate links change when blocking on more or less variables.
#
indexerBL = rl.BlockIndex(on=['state', 'city'])
candidate_links = indexerBL.index(inventors, pi)
#
len(candidate_links)
#
candidate_links[:10]
# markdown
# Let's check the first pair of candidate links blocked on city and state: (0, 85)
#
inventors.iloc[0]
#
pi.iloc[85]
#
# Now, in addition to blocking on city and state, we can also try blocking on first name
indexerBL = rl.BlockIndex(on=['name_first','city', 'state'])
candidate_links = indexerBL.index(inventors, pi)
#
len(candidate_links)
# markdown
# ## Record Comparison
#
# After you have created a set of candidate links, you’re ready to begin comparing the records associated with each candidate link. In `recordlinkage` package you must initiate a Compare object prior to performing any comparison functionality between records. This object stores both dataframes, the candidate links, and a vector containing comparison results. Further, the Compare object contains the methods for performing comparisons. The code block below initializes the comparison object.
#
# Initiate compare object
compare_cl = rl.Compare()
# markdown
# Currently there are five specific comparison methods within recordlinkage: `Compare.exact()`, `Compare.string()`, `Compare.numeric()`, `Compare.geo()`, and `Compare.date()`.
#
# The `Compare.exact()` method is simple: if two values are an exact match a comparison score of 1 is returned, otherwise 0 is retured.
#
# The `Compare.string()` method is a bit more complicated and generates a score based on well-known string-comparison algorithms (for this example, Levenshtein or Jaro Winkler).
# markdown
# The Python `recordlinkage` toolkit uses the `jellyfish` package for the Jaro, Jaro-Winkler, Levenshtein and Damerau-Levenshtein algorithms: https://jellyfish.readthedocs.io/en/latest/comparison.html.
# markdown
# There can be a large difference in the performance of different string comparison algorithms. The Jaro and Jaro-Winkler methods are faster than the Levenshtein distance and much faster than the Damerau-Levenshtein distance.
# markdown
# String similarity measures and phonetic encodings are computationally expensive. After phonetic encoding of the string variables, exact comparing can be used instead of computing the string similarity of all record pairs. If the number of candidate pairs is much larger than the number of records in both datasets together, then consider using phonetic encoding of string variables instead of string comparison.
# markdown
# > Choose and compare only informative variables: not all variables may be worth comparing in a record linkage. Some variables do not discriminate the links of the non-links or do have only minor effects. These variables can be excluded. Only informative variables should be included.
# markdown
# For this example, Jaro-Winkler distance is used (specifically developed with record linkage applications in mind, faster to compute) - words with more characters in common have a higher Jaro-Winkler value than those with fewer characters in common. The Jaro–Winkler distance gives more favorable ratings to strings that match from the beginning. The output value is normalized to fall between 0 (complete dissimilar strirngs) and 1 (exact match on strings).
# markdown
# As you remember, we already did an exact matching on `city` and `state`, when we did the blocking above and created the candidate links.
#
# We will use the string method to compare the organization names and their phonetic transcriptions.
#
# We need to specify the respective columns with organization names in both datasets, the method, and the threshold. In this case, for all strings that have more than 85% in similarity, according to the Jaro-Winkler distance, a 1 will be returned, and otherwise 0.
#
# Initiate compare object
compare_cl = rl.Compare()
#
compare_cl.string('name_first', 'name_first', method='jarowinkler', threshold=0.85, label='name_first')
compare_cl.string('name_last', 'name_last', method='jarowinkler', threshold=0.85, label='name_last')

#compare_cl.exact('state', 'state', label='state')
# markdown
# The comparing of record pairs starts when the `compute` method is called.
#
indexerBL = rl.BlockIndex(on=['city', 'state'])
candidate_links = indexerBL.index(inventors, pi)
#
len(candidate_links)
#
## All attribute comparisons are stored in a DataFrame with horizontally the features and vertically the record pairs.
features = compare_cl.compute(candidate_links, inventors, pi)
#
features.head(7)
#
features.tail()
# markdown
# ### Classification
#
# Let's check how many records we get where one or more comparison attributes match.
#
## Simple Classification: Check for how many attributes records are identical by summing the comparison results.
features.sum(axis=1).value_counts().sort_index(ascending=False)
# markdown
# We can make a decision now, and consider matches all those records which matched on all attributes in our case.
#
matches = features[features.sum(axis=1) == 2]
print(len(matches))
# markdown
# Remember that for these matches we had an exact match on `city` and `state`, and more than 80% in similarity based on organization `first name` and `last name`
#
matches.head()
# markdown
# Now let's merge these matches back to original dataframes.
# markdown
# Our `matches` dataframe has MultiIndex - two indices to the left which correspond to the `inventor` table and `pi` table respectively.
#
# We can access each matching pair individually, for example, the first one:
#
matches.index[0]
# markdown
# We can also do the following: first, put all the indices for the `inventors` table.
#
matches.index[0][0]
# markdown
# We will pull all corresponding rows from the `inventors` table.
#
inventors_results = []  # Create an empty list

for match in matches.index:  # For every pair in matches (index)
    df = pd.DataFrame(inventors.loc[[match[0]]])  # Get the location in the original table, convert to dataframe
    inventors_results.append(df)
#
inventors_results[0]
# markdown
# Now we concatenate the list of dataframes into one dataframe.
#
inventors_concat = pd.concat(inventors_results)
#
inventors_concat.head()
# markdown
# We do the same for the `pi` table.
#
pi_results = []  # Create an empty list

for match in matches.index:  # For every pair in matches (index)
    df = pd.DataFrame(pi.loc[[match[1]]])  # Get the location in the original table, convert to dataframe
    pi_results.append(df)

pi_concat = pd.concat(pi_results)
#
pi_concat.head()
# markdown
# Now we need to combine two tables on the index - notice that our tables right now have indices from the original tables. We can reset the index using `.reset_index()`.
#
inventors_concat = inventors_concat.reset_index()
pi_concat = pi_concat.reset_index()
# markdown
# Now our tables have the same index on which we can combine two tables.
#
inventors_concat.head()
#
pi_concat.head()
#
# Drop the old index column
inventors_concat = inventors_concat.drop(columns=['index'])
pi_concat = pi_concat.drop(columns=['index'])
#
# Drop other not relevant columns
inventors_concat = inventors_concat.drop(columns=['inventor_country'])
pi_concat = pi_concat.drop(columns=[' ORGANIZATION_COUNTRY'])
# markdown
# Now we concatenate these two tables using `.concat()`.
#
matched = pd.concat([inventors_concat, pi_concat], axis=1)  # Specify axis=1 to concatenate horizontally    WHY ?????????????????????????????????
#
matched[14:20]




########################################################################
# %% START WORKING HERE:
########################################################################

import pandas as pd
import recordlinkage as rl

assignees = pd.read_csv('../Data/assignees_cleaned.csv')
organizations = pd.read_csv('../Data/organizations_cleaned.csv')

assignees_raw = pd.read_csv('../Data/patentsview_assignee_org_names.csv')
# organizations = pd.read_csv('../Data/organizations_cleaned.csv')

list(organizations)

print("So far we have " + str(organizations.org_name.unique().shape[0]) + " Organizations in Grant DB")
print("So far we have " + str(assignees.org_name.unique().shape[0]) + " Organizations in assignees")

#assignees =  assignees.merge(organizations, on = ['org_name','state'], how='left')
list(assignees)
list(organizations) # I assumed that assignees have a curated version of org_name
list(assignees.org_name.value_counts().sort_index().index) # This is to check if organizations are misspelled in assignees df

compare_cl = rl.Compare()
#
compare_cl.string('org_name', 'org_name', method='jarowinkler', threshold=0.85, label='name_first')
compare_cl.string('phonetic_name', 'phonetic_name', method='jarowinkler', threshold=0.85, label='name_last')

# Define the BlockIndex by state city
indexerBL = rl.BlockIndex(on=['state','city'])
organization_links = indexerBL.index(assignees, organizations)
len(organization_links)

organizations['PROJECT_ID'].nunique()
organizations.org_name.value_counts().reset_index().sort_values('org_name', ascending=False).head(5)

assignees.org_name.value_counts().reset_index().sort_values('org_name', ascending=False).head(10)

organizations_undup =  organizations.drop_duplicates(subset = ['PROJECT_ID'])

pi = pi.merge(organizations_undup, on=['PROJECT_ID'])

pi['flag'] = 1

pi[[' CONTACT_PI_PROJECT_LEADER', 'org_name']].value_counts()
pi_t5 = pi[' CONTACT_PI_PROJECT_LEADER'].value_counts().reset_index().sort_values(' CONTACT_PI_PROJECT_LEADER', ascending=False).head(5)
pi_t5
list(pi_t5['index'])

pi_t5.merge(pi[[' CONTACT_PI_PROJECT_LEADER','org_name']], left_on='index', right_on=' CONTACT_PI_PROJECT_LEADER').drop_duplicates([' CONTACT_PI_PROJECT_LEADER_x'])




features = compare_cl.compute(organization_links, assignees, organizations)

features.head(7)

# ### Classification
#
# Let's check how many records we get where one or more comparison attributes match.
#
## Simple Classification: Check for how many attributes records are identical by summing the comparison results.
features.sum(axis=1).value_counts().sort_index(ascending=False)
# markdown
# We can make a decision now, and consider matches all those records which matched on all attributes in our case.
#
matches = features[features.sum(axis=1) == 2]
print(len(matches))
# markdown
# Remember that for these matches we had an exact match on `city` and `state`, and more than 80% in similarity based on organization `first name` and `last name`
#
matches.head()
# markdown
# Now let's merge these matches back to original dataframes.
# markdown
# Our `matches` dataframe has MultiIndex - two indices to the left which correspond to the `inventor` table and `pi` table respectively.
#
# We can access each matching pair individually, for example, the first one:
#
matches.index[0]
# markdown
# We can also do the following: first, put all the indices for the `inventors` table.
#
matches.index[0][0]
# markdown
# We will pull all corresponding rows from the `inventors` table.
#
inventors_results = []  # Create an empty list

for match in matches.index:  # For every pair in matches (index)
    df = pd.DataFrame(inventors.loc[[match[0]]])  # Get the location in the original table, convert to dataframe
    inventors_results.append(df)
#
inventors_results[0]
# markdown
# Now we concatenate the list of dataframes into one dataframe.
#
inventors_concat = pd.concat(inventors_results)
#
inventors_concat.head()
# markdown
# We do the same for the `pi` table.
#
pi_results = []  # Create an empty list

for match in matches.index:  # For every pair in matches (index)
    df = pd.DataFrame(pi.loc[[match[1]]])  # Get the location in the original table, convert to dataframe
    pi_results.append(df)

pi_concat = pd.concat(pi_results)
pi_concat.head()
inventors_concat = inventors_concat.reset_index()
pi_concat = pi_concat.reset_index()
inventors_concat.head()
pi_concat.head()
inventors_concat = inventors_concat.drop(columns=['index'])
pi_concat = pi_concat.drop(columns=['index'])
inventors_concat = inventors_concat.drop(columns=['inventor_country'])
pi_concat = pi_concat.drop(columns=[' ORGANIZATION_COUNTRY'])
matched = pd.concat([inventors_concat, pi_concat], axis=1)  # Specify axis=1 to concatenate horizontally    WHY

# markdown
# ## References and Further Readings
#
# ### Record Linkage
#
# Lane, Julia, Ian Foster, Rayid Ghani, Ron S. Jarmin, Frauke Kreuter (editors), Big Data and Social Science: A Practical Guide to Methods and Tools, Chapman and Hall/CRC Press, 2016. https://coleridge-initiative.github.io/big-data-and-social-science/chap-link.html
#
# ### Record Linkage Python package
# * `recordlinkage` Python package: https://recordlinkage.readthedocs.io/en/latest/index.html
#     - Comparing records: https://recordlinkage.readthedocs.io/en/latest/ref-compare.html
#     - Classification:
#         - https://recordlinkage.readthedocs.io/en/latest/ref-classifiers.html,
#         - https://recordlinkage.readthedocs.io/en/latest/notebooks/classifiers.html
#
# ### String Comparators
#
# * GitHub page of `jellyfish`: https://github.com/jamesturk/jellyfish
# * Descriptions of distances in `jellyfish`: https://jellyfish.readthedocs.io/en/latest/comparison.html
# * Different distances that measure the differences between strings:
#     - Levenshtein distance: https://en.wikipedia.org/wiki/Levenshtein_distance
#     - Damerau–Levenshtein distance: https://en.wikipedia.org/wiki/Damerau%E2%80%93Levenshtein_distance
#     - Jaro–Winkler distance: https://en.wikipedia.org/wiki/Jaro%E2%80%93Winkler_distance
#     - Hamming distance: https://en.wikipedia.org/wiki/Hamming_distance
#     - Match rating approach: https://en.wikipedia.org/wiki/Match_rating_approach
#
# ### Fellegi-Sunter Record Linkage
#
# * Introduction to Probabilistic Record Linkage: http://www.bristol.ac.uk/media-library/sites/cmm/migrated/documents/problinkage.pdf
# * Paper Review: https://www.cs.umd.edu/class/spring2012/cmsc828L/Papers/HerzogEtWires10.pdf
#

#
