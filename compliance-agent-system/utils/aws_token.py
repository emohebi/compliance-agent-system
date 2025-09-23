import boto3

def get_session_token_with_mfa(mfa_serial_number, mfa_token):
    """
    Gets a session token with MFA credentials and uses the temporary session
    credentials to list Amazon S3 buckets.
    
    :param mfa_serial_number: The serial number of the MFA device. For a virtual MFA
                              device, this is an Amazon Resource Name (ARN).
    :param mfa_token: A time-based, one-time password issued by the MFA device.
    """
    # Create STS client
    sts_client = boto3.client('sts')
    
    # Get session token with MFA
    if mfa_serial_number is not None:
        response = sts_client.get_session_token(
            SerialNumber=mfa_serial_number, 
            TokenCode=mfa_token,
            DurationSeconds=3600  # 1 hour
        )
    else:
        response = sts_client.get_session_token()
    
    # Extract temporary credentials
    temp_credentials = response["Credentials"]
    
    return temp_credentials


def get_boto3_client_with_temp_credentials():
 
    import boto3

    # Method 1: Use configured credentials
    client = boto3.client('sts')

    # Method 2: Explicitly specify credentials
    client = boto3.client(
        'sts',
        aws_access_key_id='none',
        aws_secret_access_key='none',
        region_name='none'
    )

    try:
        response = client.get_caller_identity()
        print("Credentials are valid:", response)
    except Exception as e:
        print("Error:", e)


if __name__ == "__main__":
    import boto3

# Method 1: Specify profile when creating session
    session = boto3.Session(profile_name='myprofile')
    sts_client = session.client('sts')

    # Method 2: Specify profile when creating client
    sts_client = boto3.client('sts', profile_name='myprofile')

    # Get session token using specific profile
    response = sts_client.get_session_token()

    # Example MFA details (replace with actual values)
    # mfa_serial = "arn:aws:iam::624136282593:user/emohebi"
    # mfa_code = "251437"  # Replace with the current MFA code from your device
    # # Example usage
    # credentials = get_session_token_with_mfa(
    #     mfa_serial_number=mfa_serial,
    #     mfa_token=mfa_code  # Replace with actual MFA code 
    # )
    # print("Temporary Credentials:", credentials)
    get_boto3_client_with_temp_credentials()

    