from typing import Union, Dict

'''
https://gist.github.com/raphaelgabbarelli/bc5a41d93789046f9c71e4685f1463e7
https://www.youtube.com/watch?v=CFEYKrP0vxs
'''
import boto3
import base64

class AwsKmsUtil(object):
    def __init__(
                    self, 
                    key_id : str,
                    profile_name : Union[str,None] = None
                ):
        self.key_id = key_id
        aws_session = boto3.session.Session(profile_name=profile_name) # type: ignore "session" is not a known attribute of module "boto3"
        self.aws_kms_client = aws_session.client('kms')
    
    def encrypt(self, plaintext : str) -> bytes:
        encrypted = self.aws_kms_client.encrypt(KeyId=self.key_id, Plaintext=plaintext)
        encrypted : bytes = base64.b64encode(encrypted['CiphertextBlob']) # type: ignore
        return encrypted

    def decrypt(self, encrypted : bytes) -> str:
        decrypted : str = self.aws_kms_client.decrypt(CiphertextBlob=base64.b64decode(encrypted)) # type: ignore Cannot access attribute "ams_kms_client" for class "AwsKmsUtil*
        return decrypted['Plaintext'].decode('utf-8') # type: ignore

if __name__ == "__main__":
    '''
    From command line, run 'aws configure' with IAM user's Access key ID and Secret access key. (Assume you have awscli installed)
        aws configure
        AWS Access Key ID [****************ABCD]: <-- ***ABCD is your IAM user 
        AWS Secret Access Key [****************xxx]: <-- xxx is password to your IAM user
        Default region name [us-east-1]: <-- Region need be where your KMS key resides!
        Default output format [None]: 
    
    Remember that when you create your KMS Key, you need to grant permission of the key newly created key to IAM user (This is done on KMS side, not IAM).
    '''
    key_id : str = "" # Enter your KMS key ID here. You'd find it from under AWS > KMS > Customer managed keys
    original : str = "some secret"
    
    aws_kms = AwsKmsUtil(key_id=key_id, profile_name=None)
    encrypted = aws_kms.encrypt(original)
    decrpted = aws_kms.decrypt(encrypted)

    print(f"original: {original}, encrypted: {encrypted.decode('utf-8')}, decrpted: {decrpted}") # type: ignore