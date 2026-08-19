"""ChromaVectorStore vendorizzato da talk-docs @6dd976c.

Modifiche del vendoring (spec §3):
- Niente ``from app.config import settings``: il client Chroma (o la ``persist_dir``) e
  l'``EmbeddingService`` sono **iniettati** dal costruttore.
- **Chunking interna NEUTRALIZZATA**: nel PoC il chunking lo possiede l'ingest (C1,
  token-based). ``add_documents`` **non** chiama più ``chunk_text``: pretende chunk già
  pronti (``document.chunks`` non vuoto) e li indicizza così come sono. Gli embedding
  arrivano dai chunk se presenti, altrimenti sono calcolati dall'``EmbeddingService``
  iniettato (mai da un ``OpenAIEmbeddingFunction`` letto da settings).
"""

import json
from typing import Any

import chromadb
from chromadb.config import Settings as ChromaSettings

from ..models.document import Document, DocumentMetadata, SearchResult
from ..services.embeddings import EmbeddingService
from .base import VectorStore


class ChromaVectorStore(VectorStore):
    """ChromaDB implementation of vector store (embedding iniettato)."""

    def __init__(
        self,
        collection_name: str,
        embedding_service: EmbeddingService,
        client: chromadb.ClientAPI | None = None,
        persist_dir: str | None = None,
    ):
        if client is not None:
            self.client = client
        else:
            if not persist_dir:
                raise ValueError("ChromaVectorStore richiede 'client' oppure 'persist_dir'")
            self.client = chromadb.PersistentClient(
                path=persist_dir, settings=ChromaSettings(anonymized_telemetry=False)
            )

        # Nessuna embedding_function: passiamo sempre embedding espliciti (custom).
        self.collection = self.client.get_or_create_collection(collection_name)
        self.embedding_service = embedding_service

    async def add_documents(self, documents: list[Document]) -> list[str]:
        """Add pre-chunked documents to ChromaDB (chunking interna neutralizzata)."""
        if not documents:
            return []

        added_ids = []

        for document in documents:
            try:
                # Chunking di C1 obbligatorio: qui non si chunk-a più.
                if not document.chunks:
                    raise ValueError(
                        f"Documento {document.source_id}: nessun chunk. Il chunking lo possiede "
                        "l'ingest (C1); ChromaVectorStore indicizza solo chunk già pronti."
                    )

                chunk_texts = [chunk.content for chunk in document.chunks]

                # Embedding: usa quelli già calcolati dai chunk se presenti, altrimenti
                # calcola con l'EmbeddingService iniettato.
                if all(chunk.embedding is not None for chunk in document.chunks):
                    embeddings = [chunk.embedding for chunk in document.chunks]
                else:
                    embeddings = await self.embedding_service.get_embeddings(chunk_texts)

                chunk_ids = []
                chunk_contents = []
                chunk_embeddings = []
                chunk_metadatas = []

                for i, chunk in enumerate(document.chunks):
                    chunk_ids.append(chunk.chunk_id)
                    chunk_contents.append(chunk.content)
                    chunk_embeddings.append(embeddings[i])

                    metadata = {
                        "source_id": document.source_id,
                        "start_index": chunk.start_index,
                        "end_index": chunk.end_index,
                        "document_metadata": document.metadata.model_dump_json(),
                        **chunk.metadata,
                    }
                    chunk_metadatas.append(metadata)

                self.collection.add(
                    ids=chunk_ids, documents=chunk_contents, embeddings=chunk_embeddings, metadatas=chunk_metadatas
                )

                added_ids.append(document.source_id)

            except Exception as e:
                print(f"Error adding document {document.source_id}: {str(e)}")
                continue

        return added_ids

    async def update_document(self, doc_id: str, document: Document) -> bool:
        """Update a document in ChromaDB."""
        try:
            await self.delete_documents([doc_id])
            result = await self.add_documents([document])
            return len(result) > 0
        except Exception as e:
            print(f"Error updating document {doc_id}: {str(e)}")
            return False

    async def delete_documents(self, doc_ids: list[str]) -> bool:
        """Delete documents from ChromaDB."""
        try:
            for doc_id in doc_ids:
                results = self.collection.get(where={"source_id": doc_id})
                if results["ids"]:
                    self.collection.delete(ids=results["ids"])
            return True
        except Exception as e:
            print(f"Error deleting documents {doc_ids}: {str(e)}")
            return False

    async def search(
        self, query: str, filters: dict[str, Any], top_k: int, min_relevance_score: float = 0.0
    ) -> list[SearchResult]:
        """Search for similar documents in ChromaDB."""
        try:
            query_embedding = await self.embedding_service.get_single_embedding(query)

            where_clause = {}
            if filters:
                if "source_id" in filters:
                    where_clause["source_id"] = filters["source_id"]

            results = self.collection.query(
                query_embeddings=[query_embedding], n_results=top_k, where=where_clause if where_clause else None
            )

            search_results = []

            if results["ids"] and results["ids"][0]:
                for i in range(len(results["ids"][0])):
                    metadata_json = results["metadatas"][0][i].get("document_metadata", "{}")
                    try:
                        doc_metadata_dict = json.loads(metadata_json)
                        doc_metadata = DocumentMetadata(**doc_metadata_dict)
                    except (json.JSONDecodeError, ValueError):
                        doc_metadata = DocumentMetadata(
                            source_id=results["metadatas"][0][i].get("source_id", "unknown"),
                            name="unknown",
                            path="unknown",
                            size=0,
                            mime_type="unknown",
                            author="unknown",
                            created_date="1970-01-01T00:00:00Z",
                            modified_date="1970-01-01T00:00:00Z",
                        )

                    distance = results["distances"][0][i]
                    similarity_score = max(0.0, (2.0 - distance) / 2.0)

                    if similarity_score < min_relevance_score:
                        continue

                    search_result = SearchResult(
                        chunk_id=results["ids"][0][i],
                        content=results["documents"][0][i],
                        score=similarity_score,
                        document_metadata=doc_metadata,
                        chunk_metadata={
                            k: v for k, v in results["metadatas"][0][i].items() if k != "document_metadata"
                        },
                    )
                    search_results.append(search_result)

            return search_results

        except Exception as e:
            print(f"Error searching: {str(e)}")
            return []

    async def get_document_metadata(self, doc_id: str) -> dict[str, Any] | None:
        """Get metadata for a specific document."""
        try:
            results = self.collection.get(where={"source_id": doc_id}, limit=1)
            if results["metadatas"]:
                metadata_json = results["metadatas"][0].get("document_metadata", "{}")
                try:
                    return json.loads(metadata_json)
                except json.JSONDecodeError:
                    return None
            return None
        except Exception as e:
            print(f"Error getting document metadata for {doc_id}: {str(e)}")
            return None

    def get_all_document_ids(self) -> list[str]:
        """Get all unique document IDs in the vector store."""
        try:
            results = self.collection.get()
            unique_docs = set()
            if results["metadatas"]:
                for metadata in results["metadatas"]:
                    source_id = metadata.get("source_id")
                    if source_id:
                        unique_docs.add(source_id)
            return list(unique_docs)
        except Exception as e:
            print(f"Error getting all document IDs: {str(e)}")
            return []

    def get_stats(self) -> dict[str, Any]:
        """Get statistics about the vector store."""
        try:
            count = self.collection.count()
            unique_docs = self.get_all_document_ids()
            return {"total_chunks": count, "unique_documents": len(unique_docs)}
        except Exception as e:
            print(f"Error getting stats: {str(e)}")
            return {"total_chunks": 0, "unique_documents": 0}
