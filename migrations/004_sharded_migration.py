from qdrant_client import models, QdrantClient

NEW_COLLECTION_NAME = "dobby-be-springworks-collection-sharded"
SOURCE_COLLECTION_NAME = "dobby-springworks-be-collection"


def create_sharded_collection(
    client: QdrantClient,
    new_collection_name: str,
    shard_number: int = 6,
    replication_factor: int = 2,
):
    client.create_collection(
        collection_name=new_collection_name,
        vectors_config=models.VectorParams(
            size=3072, distance=models.Distance.COSINE, on_disk=True
        ),
        hnsw_config=models.HnswConfigDiff(
            payload_m=16,
            m=0,
        ),
        quantization_config=models.BinaryQuantization(
            binary=models.BinaryQuantizationConfig(
                always_ram=False,
            ),
        ),
        shard_number=shard_number,
        replication_factor=replication_factor,
    )


def add_payload_indexes(
    client: QdrantClient,
    collection_name: str,
    indices: list = [
        "organisation_id",
        "document_id",
        "ref_doc_id",
        "doc_id",
        "provider",
        "category",
        "res_name",
    ],
):
    for index in indices:
        client.create_payload_index(
            collection_name=collection_name,
            field_name=index,
            field_schema=models.PayloadSchemaType.KEYWORD,
        )


def migrate_points(
    client: QdrantClient,
    src_collection_name: str,
    dest_collection_name: str,
    batch_size: int = 100,
) -> None:
    records, next_offset = client.scroll(
        src_collection_name, limit=2, with_vectors=True
    )
    client.upload_records(dest_collection_name, records)
    print("Migration started")
    while next_offset is not None:
        records, next_offset = client.scroll(
            src_collection_name, offset=next_offset, limit=batch_size, with_vectors=True
        )
        client.upload_records(dest_collection_name, records, wait=True)
        print(f"moved {len(records)} records. {next_offset=}")

    source_client_vectors_count = client.get_collection(
        src_collection_name
    ).vectors_count
    dest_client_vectors_count = client.get_collection(
        dest_collection_name
    ).vectors_count

    assert (
        source_client_vectors_count == dest_client_vectors_count
    ), f"Migration failed, vectors count are not equal: source vector count {source_client_vectors_count}, dest vector count {dest_client_vectors_count}"

    print("Migration succeeded")


def forward(client):
    create_sharded_collection(client=client, new_collection_name=NEW_COLLECTION_NAME)
    add_payload_indexes(client=client, collection_name=NEW_COLLECTION_NAME)
    migrate_points(
        client=client,
        src_collection_name=SOURCE_COLLECTION_NAME,
        dest_collection_name=NEW_COLLECTION_NAME,
    )


def backward(client):
    # client.delete_collection(NEW_COLLECTION_NAME)
    print("Your code to rollback the migration here")
