from qdrant_client import models


def forward(client):
    # Your migration logic here
    client.update_collection(
        collection_name="collection-name",
        # vectors_config=models.VectorParams(
        #     size=3072, distance=models.Distance.COSINE, on_disk=True
        # ),
        hnsw_config=models.HnswConfigDiff(
            payload_m=16,
            m=0,
            on_disk=True,
        ),
        quantization_config=models.BinaryQuantization(
            binary=models.BinaryQuantizationConfig(
                always_ram=False,  # This mode allows to achieve the smallest memory footprint, but at the cost of the search speed.
            ),
        ),
    )
    indices = [
        "organisation_id",
        "document_id",
        "ref_doc_id",
        "doc_id",
        "provider",
        "category",
    ]

    for index in indices:
        client.create_payload_index(
            collection_name="collection-name",
            field_name=index,
            field_schema=models.PayloadSchemaType.KEYWORD,
        )


def backward(client):
    # Logic to undo the migration if needed
    # client.delete_collection("new_collection")
    print("Your code to rollback the migration here")
