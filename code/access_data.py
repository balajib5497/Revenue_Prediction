from google.cloud import bigquery
from google.oauth2 import service_account

from util import *

credentials = service_account.Credentials.from_service_account_file(
    '../dummy project-1e32558e44e7.json')
project_id = 'dummy-project-220805'

client = bigquery.Client(credentials= credentials,project=project_id)

query_job = client.query("""
  SELECT *
  FROM Analytics.ga_sessions_20170801""")

results = query_job.result()

df = results.to_dataframe()

cleaned = clean_json_blobs(df, is_train=True)
