from qdrant_client import QdrantClient
from qdrant_client.http.models.models import PayloadSchemaType


def forward(client: QdrantClient):
    client.create_payload_index(
        collection_name="dobby-be-springworks-collection-sharded",
        field_name="chunk_hash",
        field_schema=PayloadSchemaType.KEYWORD,
    )

    client.create_payload_index(
        collection_name="dobby-be-springworks-collection-sharded",
        field_name="version",
        field_schema=PayloadSchemaType.KEYWORD,
    )


def backward(client: QdrantClient):
    client.delete_payload_index(
        collection_name="dobby-springworks-be-sharded",
        field_name="chunk_hash",
    )

    client.delete_payload_index(
        collection_name="dobby-springworks-be-sharded",
        field_name="version",
    )
