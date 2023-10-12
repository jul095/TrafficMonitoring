import os
from azure.storage.blob import BlobServiceClient


def upload_file(local_path, file_name_to_upload):
    #print("local_path", local_path)
    #print("file_name_to_upload", file_name_to_upload)

    try:
        primary_connection_string = os.getenv('PRIMARY_CONNECTION_STRING')
        container_name = os.getenv('BLOB_CONTAINER_NAME', 'files')
        blob_path = os.getenv('BLOB_PATH', '20211213-15:59:44:368-273083b6')

        #print("BLOB_CONTAINER_NAME", container_name)
        #print("BLOB_PATH", pfad)

        # Create the BlobServiceClient object which will be used to create a container client
        blob_service_client = BlobServiceClient.from_connection_string(primary_connection_string)
        upload_file_path = os.path.join(local_path, file_name_to_upload)

        blob_client = blob_service_client.get_blob_client(container=container_name, blob=os.path.join(blob_path, file_name_to_upload))


        print("\nUploading to Azure Storage as blob:\n\t" + file_name_to_upload)

        with open(upload_file_path, "rb") as data:
            blob_client.upload_blob(data, overwrite=True)


    except Exception as ex:
        print('Exception:')
        print(ex.with_traceback())
    return file_name_to_upload


#upload_file_data("/home/felix/PycharmProjects/TrafficMonitoring/traffic_monitoring", "teamcloud_uploadtet.test")

