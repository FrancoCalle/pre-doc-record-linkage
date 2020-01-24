import pandas as pd
import numpy as np
# record linkage and preprocessing
import recordlinkage as rl
from recordlinkage.preprocessing import clean, phonetic

#inventors = pd.read_csv('../Data/patentsview_inventors.csv')
assignees = pd.read_csv('../Data/patentsview_assignee_org_names.csv')
organizations = pd.read_csv('../Data/fedreporter_org_names_grants.csv')


#Do cleaning for assignee
assignees.head()
assignees['organizations_clean'] = clean(assignees['organization'])
assignees['organization'] = assignees['organizations_clean'].str.replace(' ','')
assignees["organization_phonetic"] = phonetic(assignees["organization"], method="nysiis")
assignees  = assignees.loc[assignees.org_country == "US",:]

#Do cleaning for organization
organizations[' ORGANIZATION_NAME_CLEAN'] = clean(organizations[' ORGANIZATION_NAME'])
organizations['organization'] = organizations[' ORGANIZATION_NAME_CLEAN'].str.replace(' ','')
organizations['organization_phonetic'] = phonetic(organizations["organization"], method="nysiis")
organizations  = organizations.loc[organizations[' ORGANIZATION_COUNTRY'] == "UNITED STATES",:]


print(assignees.shape)
print(assignees.patent_id.drop_duplicates().shape)

#Explore stuff
assignees.head()
organizations.head()



# rightdb = organizations[' ORGANIZATION_STATE'].unique()
# leftdb = assignees.org_state.unique()
organizations[' ORGANIZATION_NAME'].unique().shape
assignees['organization'].unique().shape




organizations['initial_org'] = list(map(lambda x: x[1]))
assignees['initial_org']
organizationsMerged = organizations.merge(assignees, left_on=['organization', ' ORGANIZATION_STATE'], right_on=['organization', 'org_state'], how='left', indicator=True)

organizationsMerged = organizations.merge(assignees, left_on=['organization', ' ORGANIZATION_STATE',' ORGANIZATION_CITY'], right_on=['organization', 'org_state','org_city'], how='left', indicator=True)

organizationsMerged

assignees.organization.unique().shape

assignees.head()
organizations.head()

organizations.shape
assignees.shape
organizationsMerged.shape

organizationsMerged['PROJECT_ID'].unique()

organizationsMerged.shape
(organizationsMerged._merge == 'both').mean()


organizationsMerged.drop_duplicates(['PROJECT_ID']).patent_id.isna().mean()

organizationsMerged.patent_id.isna().mean()

leftdb.shape
rightdb.shape

matchedDataset = pd.DataFrame()
for list()


#Explore stuff



organizations.head()
organizations.shape


#Match by ORGANIZATION STATE AND MATCH
