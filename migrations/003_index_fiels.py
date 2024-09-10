from qdrant_client import models


def forward(client):
    indices = ["res_name"]

    for index in indices:
        client.create_payload_index(
            collection_name="dobby-springworks-be-collection",
            field_name=index,
            field_schema=models.PayloadSchemaType.KEYWORD,
        )


def backward(client):
    # Logic to undo the migration if needed
    # client.delete_collection("new_collection")
    print("Your code to rollback the migration here")
