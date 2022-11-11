import pathlib
import pandas as pd
import dropbox
from dropbox.exceptions import AuthError
from dropbox import DropboxOAuth2FlowNoRedirect

import os
from tqdm import tqdm
import sys
import argparse

parser = argparse.ArgumentParser(description="Folder")
parser.add_argument("--Folder",type=str)
parser.add_argument("--MotherRoot",type=str)
parser.add_argument("--DaughterRoot",type=str)
parser.add_argument("--access_token",type=str)
parser.add_argument("--refresh_token",type=str)

args = parser.parse_args()


Folder = args.Folder
MotherRoot=args.MotherRoot
DaughterRoot=args.DaughterRoot




'''
Populate your app key in order to run this locally
'''
APP_KEY = "xfyyilhghe0tjgx"
APP_SECRET = "i4usp8n8zd1m1za"
# DROPBOX_ACCESS_TOKEN = 'sl.BSp2M7eldpMjchM7fVxec79meoXB3SWZyot82JJeegbKLGKXzCBO7Fmzeo4oeHg32jCKp1pAbyCOTF3lU2jLmt52NM_cziZreHyAdAMvlV1lr-qRTD6jySXrHMl_8rLU1Egcs2Ud63zs'

access_token = args.access_token
refresh_token = args.refresh_token

with dropbox.Dropbox(oauth2_access_token=access_token,
                     oauth2_refresh_token=refresh_token,
                     app_key=APP_KEY,
                     app_secret=APP_SECRET):
    print("Successfully set up client!")


def dropbox_connect():
    """Create a connection to Dropbox."""

    try:
        dbx = dropbox.Dropbox(DROPBOX_ACCESS_TOKEN)
    except AuthError as e:
        print('Error connecting to Dropbox with access token: ' + str(e))
    return dbx


def dropbox_list_files(path):
    """Return a Pandas dataframe of files in a given Dropbox folder path in the Apps directory.
    """

    dbx = dropbox_connect()

    try:
        files = dbx.files_list_folder(path).entries
        files_list = []
        for file in files:
            if isinstance(file, dropbox.files.FileMetadata):
                metadata = {
                    'name': file.name,
                    'path_display': file.path_display,
                    'client_modified': file.client_modified,
                    'server_modified': file.server_modified
                }
                files_list.append(metadata)

        df = pd.DataFrame.from_records(files_list)
        # return df.sort_values(by='server_modified', ascending=False)
        print(df)
        # return df

    except Exception as e:
        print('Error getting list of files from Dropbox: ' + str(e))


def dropbox_download_file(dropbox_file_path, local_file_path):
    """Download a file from Dropbox to the local machine."""

    try:
        dbx = dropbox_connect()

        with open(local_file_path, 'wb') as f:
            metadata, result = dbx.files_download(path=dropbox_file_path)
            f.write(result.content)
    except Exception as e:
        print('Error downloading file from Dropbox: ' + str(e))


def dropbox_upload_file(local_path, local_file, dropbox_file_path):
    """Upload a file from the local machine to a path in the Dropbox app directory.

    Args:
        local_path (str): The path to the local file.
        local_file (str): The name of the local file.
        dropbox_file_path (str): The path to the file in the Dropbox app directory.

    Example:
        dropbox_upload_file('.', 'test.csv', '/stuff/test.csv')

    Returns:
        meta: The Dropbox file metadata.
    """

    try:
        dbx = dropbox_connect()

        local_file_path = pathlib.Path(local_path) / local_file

        with local_file_path.open("rb") as f:
            meta = dbx.files_upload(
                f.read(), dropbox_file_path, mode=dropbox.files.WriteMode("overwrite"))

            return meta
    except Exception as e:
        print('Error uploading file to Dropbox: ' + str(e))


def dropbox_upload_folder(local_folder_path, Dropbox_folder_path):
    try:
        dbx = dropbox_connect()

        # enumerate local files recursively
        for root, dirs, files in os.walk(local_folder_path):

            for filename in files:

                # construct the full local path
                local_path = os.path.join(root, filename)

                # construct the full Dropbox path
                relative_path = os.path.relpath(local_path, local_folder_path)
                dropbox_path = os.path.join(Dropbox_folder_path, relative_path)

                with open(local_path, "rb") as f:
                    meta = dbx.files_upload(
                        f.read(), dropbox_path, mode=dropbox.files.WriteMode("overwrite"))
    except Exception as e:
        print('Error uploading file to Dropbox: ' + str(e))


def dropbox_upload_folder_LARGE(
    local_folder_path,
    Dropbox_folder_path,
    timeout=900,
    chunk_size=4 * 1024 * 1024,
):
    try:
        dbx = dropbox.Dropbox(DROPBOX_ACCESS_TOKEN, timeout=timeout)
        # enumerate local files recursively
        for root, dirs, files in os.walk(local_folder_path):

            for filename in files:

                # construct the full local path
                local_path = os.path.join(root, filename)

                # construct the full Dropbox path
                relative_path = os.path.relpath(local_path, local_folder_path)
                dropbox_path = os.path.join(Dropbox_folder_path, relative_path)

                with open(local_path, "rb") as f:
                    file_size = os.path.getsize(local_path)
                    if file_size <= chunk_size:
                        print(dbx.files_upload(f.read(), dropbox_path))
                    else:
                        with tqdm(total=file_size, desc="Uploaded") as pbar:
                            upload_session_start_result = dbx.files_upload_session_start(
                                f.read(chunk_size)
                            )
                            pbar.update(chunk_size)
                            cursor = dropbox.files.UploadSessionCursor(
                                session_id=upload_session_start_result.session_id,
                                offset=f.tell(),
                            )
                            commit = dropbox.files.CommitInfo(
                                path=dropbox_path)
                            while f.tell() < file_size:
                                if (file_size - f.tell()) <= chunk_size:
                                    print(
                                        dbx.files_upload_session_finish(
                                            f.read(chunk_size), cursor, commit
                                        )
                                    )
                                else:
                                    dbx.files_upload_session_append(
                                        f.read(chunk_size),
                                        cursor.session_id,
                                        cursor.offset,
                                    )
                                    cursor.offset = f.tell()
                                pbar.update(chunk_size)
    except Exception as e:
        print('Error uploading file to Dropbox: ' + str(e))


def dropbox_upload_folder_LARGE_LongToken(
    local_folder_path,
    Dropbox_folder_path,
    timeout=900,
    chunk_size=4 * 1024 * 1024,
):
    try:
        dbx = dropbox.Dropbox(oauth2_access_token=access_token,
                              oauth2_refresh_token=refresh_token,
                              app_key=APP_KEY,
                              app_secret=APP_SECRET)
        # enumerate local files recursively
        for root, dirs, files in os.walk(local_folder_path):

            for filename in files:

                # construct the full local path
                local_path = os.path.join(root, filename)

                # construct the full Dropbox path
                relative_path = os.path.relpath(local_path, local_folder_path)
                dropbox_path = os.path.join(Dropbox_folder_path, relative_path)

                with open(local_path, "rb") as f:
                    file_size = os.path.getsize(local_path)
                    if file_size <= chunk_size:
                        print(dbx.files_upload(f.read(), dropbox_path))
                    else:
                        with tqdm(total=file_size, desc="Uploaded") as pbar:
                            upload_session_start_result = dbx.files_upload_session_start(
                                f.read(chunk_size)
                            )
                            pbar.update(chunk_size)
                            cursor = dropbox.files.UploadSessionCursor(
                                session_id=upload_session_start_result.session_id,
                                offset=f.tell(),
                            )
                            commit = dropbox.files.CommitInfo(
                                path=dropbox_path)
                            while f.tell() < file_size:
                                if (file_size - f.tell()) <= chunk_size:
                                    print(
                                        dbx.files_upload_session_finish(
                                            f.read(chunk_size), cursor, commit
                                        )
                                    )
                                else:
                                    dbx.files_upload_session_append(
                                        f.read(chunk_size),
                                        cursor.session_id,
                                        cursor.offset,
                                    )
                                    cursor.offset = f.tell()
                                pbar.update(chunk_size)
    except Exception as e:
        print('Error uploading file to Dropbox: ' + str(e))


MotherFolder=MotherRoot+Folder+"/"
DaugherFolder=DaughterRoot+Folder+"/"

dropbox_upload_folder_LARGE_LongToken(MotherFolder,DaugherFolder)
