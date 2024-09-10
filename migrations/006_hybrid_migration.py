from qdrant_client import models, QdrantClient
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed

logger = logging.getLogger(__name__)
from fastembed import SparseTextEmbedding
import copy
import json

NEW_COLLECTION_NAME = "dobby-be-springworks-hybrid-collection"
SOURCE_COLLECTION_NAME = "dobby-be-springworks-collection-sharded"


def create_hybrid_collection(
    client: QdrantClient,
    new_collection_name: str,
    shard_number: int = 6,
    replication_factor: int = 2,
):
    client.create_collection(
        collection_name=new_collection_name,
        vectors_config={
            "text-dense": models.VectorParams(
                size=3072, distance=models.Distance.COSINE, on_disk=True
            )
        },
        sparse_vectors_config={
            "text-sparse": models.SparseVectorParams(
                index=models.SparseIndexParams(
                    on_disk=True,
                )
            )
        },
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
        "chunk_hash",
        "version",
    ],
):
    for index in indices:
        client.create_payload_index(
            collection_name=collection_name,
            field_name=index,
            field_schema=models.PayloadSchemaType.KEYWORD,
        )


def generate_and_add_sparse_vector_to_records(records):
    SPARSE_MODEL = "Qdrant/bm42-all-minilm-l6-v2-attentions"
    updated_records = []

    for record in records:
        node_content = record.payload["_node_content"]
        parsed_content = json.loads(node_content)
        text = parsed_content.get("text", "")
        sparse_model = SparseTextEmbedding(model_name=SPARSE_MODEL)
        sparse_embedding = list(sparse_model.embed(str(text)))

        updated_record = copy.deepcopy(record)

        updated_record.vector = {
            "text-dense": record.vector,
            "text-sparse": sparse_embedding,
        }

        updated_records.append(updated_record)

    return updated_records


def migrate_points(
    client: QdrantClient,
    src_collection_name: str,
    dest_collection_name: str,
    batch_size: int = 100,
    next_offset_manual: str = None,
    max_workers: int = 4,  # Number of parallel workers
) -> None:
    def upload_batch(records, next_offset):
        try:
            updated_records = generate_and_add_sparse_vector_to_records(records)
            client.upload_records(dest_collection_name, updated_records, wait=False)
            logger.info(f"Moved {len(records)} records. {next_offset=}")
        except Exception as e:
            logger.error(f"Failed to upload records batch: {e}")
            raise e

    try:
        _next_offset = next_offset_manual
        records, next_offset = client.scroll(
            src_collection_name, limit=2, with_vectors=True, next_offset=_next_offset
        )
        upload_batch(records, next_offset)
        logger.info("Migration started")

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = []

            while next_offset is not None:
                records, next_offset = client.scroll(
                    src_collection_name,
                    offset=next_offset,
                    limit=batch_size,
                    with_vectors=True,
                )
                _next_offset = next_offset
                futures.append(executor.submit(upload_batch, records, next_offset))

            # Ensure all futures are completed
            for future in as_completed(futures):
                try:
                    future.result()  # This will raise an exception if the future has one
                except Exception as e:
                    logger.error(f"Exception in future: {e}")

        source_client_vectors_count = client.get_collection(
            src_collection_name
        ).vectors_count
        dest_client_vectors_count = client.get_collection(
            dest_collection_name
        ).vectors_count

        assert (
            source_client_vectors_count == dest_client_vectors_count
        ), f"Migration failed, vectors count are not equal: source vector count {source_client_vectors_count}, dest vector count {dest_client_vectors_count}"
    except Exception as e:
        logger.error(f"Exception occurred: {e}")
        migrate_points(
            client,
            src_collection_name,
            dest_collection_name,
            batch_size,
            _next_offset,
            max_workers,
        )
    logger.info("Migration succeeded")


def forward(client):
    create_hybrid_collection(client=client, new_collection_name=NEW_COLLECTION_NAME)
    add_payload_indexes(client=client, collection_name=NEW_COLLECTION_NAME)
    migrate_points(
        client=client,
        src_collection_name=SOURCE_COLLECTION_NAME,
        dest_collection_name=NEW_COLLECTION_NAME,
    )


def backward(client):
    # client.delete_collection(NEW_COLLECTION_NAME)
    print("Your code to rollback the migration here")
