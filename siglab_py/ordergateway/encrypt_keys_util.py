from typing import Union
from siglab_py.util.aws_util import AwsKmsUtil

'''
From command line, run 'aws configure' with IAM user's Access key ID and Secret access key. (Assume you have awscli installed)
    aws configure
    AWS Access Key ID [****************ABCD]: <-- ***ABCD is your IAM user 
    AWS Secret Access Key [****************xxx]: <-- xxx is password to your IAM user
    Default region name [us-east-1]: <-- Region need be where your KMS key resides!
    Default output format [None]: 

Remember that when you create your KMS Key, you need to grant permission of the key newly created key to IAM user (This is done on KMS side, not IAM).
'''
key_id : Union[str, None] = None
api_key : Union[str, None] = None
secret : Union[str, None] = None
passphrase : Union[str, None] = None

print("enter key_id")
key_id = input()

print("enter apikey")
apikey = input()

print("enter secret")
secret = input()

print("enter passphrase")
passphrase = input()

aws_kms = AwsKmsUtil(key_id=key_id, profile_name=None)

apikey = aws_kms.encrypt(apikey).decode("utf-8")
secret = aws_kms.encrypt(secret).decode("utf-8")

if passphrase:
    passphrase = aws_kms.encrypt(passphrase).decode("utf-8")

print(f"apikey: {apikey}")
print(f"secret: {secret}")

if passphrase:
    print(f"passphrase: {passphrase}")
