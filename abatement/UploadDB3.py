import dropbox
from dropbox import DropboxOAuth2FlowNoRedirect

'''
Populate your app key in order to run this locally
'''
APP_KEY = "xfyyilhghe0tjgx"
APP_SECRET = "i4usp8n8zd1m1za"


auth_flow = DropboxOAuth2FlowNoRedirect(APP_KEY,
                                        consumer_secret=APP_SECRET,
                                        token_access_type='offline',
                                        scope=['account_info.write', 'files.metadata.write', 'files.content.write', 'files.content.read', 'sharing.write', 'file_requests.write', 'contacts.write'])

authorize_url = auth_flow.start()
print("1. Go to: " + authorize_url)
print("2. Click \"Allow\" (you might have to log in first).")
print("3. Copy the authorization code.")
auth_code = input("Enter the authorization code here: ").strip()

# auth_code = "onK8w2voafAAAAAAAAAO-8WnWH7OnIPzAZN7iuZLKF0"

try:
    oauth_result = auth_flow.finish(auth_code)
    print(oauth_result.access_token)
    print(oauth_result.refresh_token)
    # Oauth token has files.metadata.read scope only
    # assert oauth_result.scope == 'files.metadata.read'
except Exception as e:
    print('Error: %s' % (e,))
    exit(1)


with dropbox.Dropbox(oauth2_access_token=oauth_result.access_token,
                     oauth2_access_token_expiration=oauth_result.expires_at,
                     oauth2_refresh_token=oauth_result.refresh_token,
                     app_key=APP_KEY,
                     app_secret=APP_SECRET):
    print("Successfully set up client!")
