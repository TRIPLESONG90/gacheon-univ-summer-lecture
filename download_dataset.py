import boto3
s3 = boto3.client(
	's3',
	aws_access_key_id="twVSwbtrHQB6vSRJd7hGpToETAdkYG68",
	aws_secret_access_key="5Zwj8gnI0S6dQhhtljG98TIjGmg4E0Wd",
    region_name='tw-001',
    endpoint_url='https://tw-001.s3.synologyc2.net'
)

s3.download_file('gacheon-univ', 'dataset.zip', 'dataset.zip')

print("download completed")

